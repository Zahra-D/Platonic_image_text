import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F

from data import BalancedUnpairedDataset, ClevrMultimodalDataset, ClevrTextTokenizer, MultimodalCollator
from models import (
    IMAGE_PRIVATE_ONLY_ROUTE_ID,
    IMAGE_ROUTE_ID,
    TEXT_PRIVATE_ONLY_ROUTE_ID,
    TEXT_ROUTE_ID,
    MultimodalMaskedTransformer,
    TriLoRALinear,
    inject_lora,
    lora_modality_context,
    parameter_counts,
    shared_activation_context,
)
from models.multimodal_transformer import gradient_reverse
from multimodal_diffusion import corrupt_batch, generate_conditioned_images, masked_loss
from shared_jepa import data2vec_hidden_loss, shared_latent_jepa_loss
from objective_gradient_diagnostics import objective_shared_gradient_metrics
from train_multimodal import (
    data2vec_tristage_multiplier,
    deterministic_subset_indices,
    load_initial_model_weights,
    modality_sub_batches,
    optimizer_groups,
    paired_t1_selection_loss,
    set_private_lora_trainable,
    update_ema_teacher,
)
from alignment_evaluation import _apply_target_mask, _conditioned_batch, _target_mask_spec
from unpaired_backtranslation import (
    IMAGE_MODALITY,
    TEXT_MODALITY,
    combine_unpaired_batches,
    translation_routes,
    unpaired_backtranslation_losses,
)


class MultimodalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        images = [{"modality": "image", "image_path": f"images/{index}.png"} for index in range(3)]
        texts = [
            {"modality": "text", "text": text}
            for text in ("red cube.", "blue sphere.", "green cylinder.")
        ]
        (self.root / "images.jsonl").write_text("\n".join(json.dumps(row) for row in images))
        (self.root / "text.jsonl").write_text("\n".join(json.dumps(row) for row in texts))
        self.pairs = [
            {"image_path": images[index]["image_path"], "caption_human": texts[index]["text"]}
            for index in range(3)
        ]
        (self.root / "pairs.jsonl").write_text("\n".join(json.dumps(row) for row in self.pairs))
        torch.save({"tokens": torch.randint(0, 8, (3, 2, 2), dtype=torch.uint16)}, self.root / "tokens.pt")
        self.tokenizer = ClevrTextTokenizer.build([row["text"] for row in texts])
        self.collator = MultimodalCollator(self.tokenizer, num_image_codes=8, max_text_length=8)

    def tearDown(self):
        self.temp.cleanup()

    def test_deterministic_subset_is_reproducible_and_rng_independent(self):
        expected = deterministic_subset_indices(20, 7, 20260826)
        torch.manual_seed(999)
        torch.rand(100)
        self.assertEqual(deterministic_subset_indices(20, 7, 20260826), expected)
        self.assertEqual(len(expected), 7)
        self.assertEqual(len(set(expected)), 7)
        self.assertNotEqual(deterministic_subset_indices(20, 7, 20260827), expected)
        self.assertEqual(deterministic_subset_indices(3, 10, None), [0, 1, 2])

    def test_paired_forward_loss_and_generation(self):
        dataset = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "paired")
        batch = self.collator([dataset[0], dataset[1]])
        image_mask = batch["eligible_mask"] & batch["modality_ids"].eq(2)
        self.assertEqual(image_mask.sum().item(), 8)
        model = MultimodalMaskedTransformer(self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0)
        corrupted, masked, t = corrupt_batch(
            batch["input_ids"], batch["eligible_mask"], batch["modality_ids"],
            self.tokenizer.mask_id, objective="image",
        )
        logits = model(corrupted, batch["attention_mask"], batch["position_ids"], batch["modality_ids"])
        loss = masked_loss(logits, batch["input_ids"], masked, t, False)
        loss.backward()
        generated = generate_conditioned_images(
            model, batch, self.collator.image_offset, 8, self.tokenizer.mask_id, num_steps=3,
        )
        self.assertEqual(tuple(generated.shape), (2, 4))
        self.assertGreaterEqual(generated.min().item(), 0)
        self.assertLess(generated.max().item(), 8)

    def test_masked_loss_supports_default_one_over_t_weighting(self):
        logits = torch.zeros((2, 1, 2))
        targets = torch.zeros((2, 1), dtype=torch.long)
        masked = torch.ones((2, 1), dtype=torch.bool)
        t = torch.tensor([0.5, 1.0])
        weighted = masked_loss(logits, targets, masked, t, weight_by_t=True)
        unweighted = masked_loss(logits, targets, masked, t, weight_by_t=False)
        self.assertTrue(torch.allclose(weighted, unweighted * 1.5))

    def test_single_pair_manifest_and_caption_field(self):
        dataset = ClevrMultimodalDataset(
            self.root, self.root / "tokens.pt", "paired",
            pair_manifest=self.root / "pairs.jsonl", caption_field="caption_human",
        )
        self.assertEqual(dataset[1]["text"], "blue sphere.")
        self.assertEqual(dataset.image_records[1]["image_path"], "images/1.png")

    def test_balanced_unpaired_is_deranged_and_split(self):
        source = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        dataset = BalancedUnpairedDataset(source, seed=13)
        self.assertEqual(len(dataset), 3)
        for text_index, image_index in zip(dataset.text_indices, dataset.image_indices):
            self.assertNotEqual(text_index, image_index)
        batch = self.collator([dataset[0], dataset[1]])
        text_batch, image_batch = batch["text_batch"], batch["image_batch"]
        self.assertTrue(text_batch["modality_ids"].eq(1).any())
        self.assertFalse(text_batch["modality_ids"].eq(2).any())
        self.assertTrue(image_batch["modality_ids"].eq(2).any())
        self.assertFalse(image_batch["modality_ids"].eq(1).any())
        self.assertTrue(text_batch["route_ids"].eq(TEXT_ROUTE_ID).all())
        self.assertTrue(image_batch["route_ids"].eq(IMAGE_ROUTE_ID).all())

    def test_balanced_unpaired_redraws_a_new_derangement_each_epoch(self):
        source = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        dataset = BalancedUnpairedDataset(source, seed=13)
        epoch_zero_pairs = dict(zip(dataset.text_indices, dataset.image_indices))
        dataset.set_epoch(1)
        epoch_one_pairs = dict(zip(dataset.text_indices, dataset.image_indices))
        self.assertTrue(all(text != image for text, image in epoch_one_pairs.items()))
        self.assertTrue(
            all(epoch_zero_pairs[text] != image for text, image in epoch_one_pairs.items())
        )

    def test_backtranslation_uses_strict_unpaired_shared_only_sources(self):
        source = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        dataset = BalancedUnpairedDataset(source, seed=13)
        batch = self.collator([dataset[0], dataset[1]])
        text_batch, image_batch = batch["text_batch"], batch["image_batch"]
        paired = combine_unpaired_batches(text_batch, image_batch, self.tokenizer.pad_id)

        text_to_image = translation_routes(paired, IMAGE_MODALITY)
        self.assertTrue(text_to_image[paired["modality_ids"].eq(TEXT_MODALITY)].eq(-1).all())
        self.assertTrue(text_to_image[paired["modality_ids"].eq(IMAGE_MODALITY)].eq(IMAGE_ROUTE_ID).all())
        image_to_text = translation_routes(paired, TEXT_MODALITY)
        self.assertTrue(image_to_text[paired["modality_ids"].eq(IMAGE_MODALITY)].eq(-1).all())
        self.assertTrue(image_to_text[paired["modality_ids"].eq(TEXT_MODALITY)].eq(TEXT_ROUTE_ID).all())

    def test_backtranslation_losses_are_differentiable_in_both_directions(self):
        source = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        dataset = BalancedUnpairedDataset(source, seed=13)
        batch = self.collator([dataset[0], dataset[1]])
        model = MultimodalMaskedTransformer(self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0)
        from models import inject_tri_lora
        inject_tri_lora(model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 2, 0.0)
        model.train()
        results = unpaired_backtranslation_losses(
            model, batch["text_batch"], batch["image_batch"], self.tokenizer,
            self.collator.image_offset, 8, batch_size=2, generation_steps=2,
            temperature=0.0, reveal_order="confidence", minimum_confidence=0.0,
            cycle_weight=1.0, alignment_weight=0.1, eps=1e-3,
            weight_by_t=False, alignment_loss_type="contrastive",
            contrastive_temperature=0.07,
        )
        self.assertEqual(set(results), {"text_to_image", "image_to_text"})
        self.assertTrue(model.training)
        for result in results.values():
            self.assertIsNotNone(result["loss"])
            self.assertTrue(torch.isfinite(result["loss"]))
            self.assertEqual(result["acceptance"], 1.0)
            self.assertIn("alignment_top1", result)
            result["loss"].backward()
        tri_modules = [module for module in model.modules() if isinstance(module, TriLoRALinear)]
        self.assertTrue(any(module.shared_B.grad is not None for module in tri_modules))
        self.assertTrue(any(module.text_B.grad is not None for module in tri_modules))
        self.assertTrue(any(module.image_B.grad is not None for module in tri_modules))

    def test_shared_token_translation_uses_condition_kv_in_selected_layer(self):
        source = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        dataset = BalancedUnpairedDataset(source, seed=13)
        batch = self.collator([dataset[0], dataset[1]])
        condition, target = batch["text_batch"], batch["image_batch"]
        model = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            use_modality_embeddings=False, shared_translation_layers=[1],
        )
        from models import inject_tri_lora
        inject_tri_lora(
            model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 2, 0.0,
            delete_base_weights=True,
        )
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, TriLoRALinear):
                    module.shared_B.normal_(std=0.02)
        target_ids = target["input_ids"].masked_fill(
            target["eligible_mask"], self.tokenizer.mask_id
        )
        logits, attention = model.forward_shared_translation(
            condition["input_ids"], condition["attention_mask"], condition["position_ids"],
            condition["modality_ids"], condition["route_ids"], target_ids,
            target["attention_mask"], target["position_ids"], target["modality_ids"],
            target["route_ids"], return_bridge_attention=True,
        )
        self.assertEqual(logits.shape[:2], target_ids.shape)
        self.assertEqual(set(attention), {1})
        self.assertEqual(attention[1].shape[:3], (2, 4, target_ids.size(1)))
        self.assertTrue(torch.isfinite(attention[1]).all())
        self.assertTrue(torch.allclose(attention[1].sum(-1), torch.ones_like(attention[1].sum(-1))))

        results = unpaired_backtranslation_losses(
            model, condition, target, self.tokenizer, self.collator.image_offset, 8,
            batch_size=2, generation_steps=2, temperature=0.0,
            reveal_order="confidence", minimum_confidence=0.0,
            cycle_weight=1.0, alignment_weight=0.0, eps=1e-3,
            weight_by_t=False,
        )
        sum(result["loss"] for result in results.values()).backward()
        self.assertTrue(any(
            bridge.out_proj.weight.grad is not None
            and bridge.out_proj.weight.grad.abs().sum().item() > 0
            for bridge in model.shared_translation_bridges.values()
        ))

    def test_soft_permutation_translation_resamples_source_shared_to_target_length(self):
        source = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        dataset = BalancedUnpairedDataset(source, seed=13)
        batch = self.collator([dataset[0], dataset[1]])
        condition, target = batch["text_batch"], batch["image_batch"]
        model = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            use_modality_embeddings=False, shared_translation_layers=[1],
            shared_translation_mode="soft_permutation",
        )
        from models import inject_tri_lora
        inject_tri_lora(
            model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 2, 0.0,
            delete_base_weights=True,
        )
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, TriLoRALinear):
                    module.shared_B.normal_(std=0.02)
            for bridge in model.shared_translation_bridges.values():
                bridge.replacement_gate.fill_(0.1)
        target_ids = target["input_ids"].masked_fill(
            target["eligible_mask"], self.tokenizer.mask_id
        )
        logits, attention = model.forward_shared_translation(
            condition["input_ids"], condition["attention_mask"], condition["position_ids"],
            condition["modality_ids"], condition["route_ids"], target_ids,
            target["attention_mask"], target["position_ids"], target["modality_ids"],
            target["route_ids"], return_bridge_attention=True,
            condition_kv_mask=condition["eligible_mask"],
        )
        bridge = model.shared_translation_bridges["1"]
        self.assertIsNone(bridge.v_proj)
        self.assertIsNone(bridge.out_proj)
        self.assertEqual(logits.shape[:2], target_ids.shape)
        self.assertEqual(attention[1].shape[2], target_ids.size(1))
        self.assertEqual(attention[1].shape[3], condition["input_ids"].size(1))
        logits.square().mean().backward()
        self.assertIsNotNone(bridge.replacement_gate.grad)
        self.assertGreater(bridge.replacement_gate.grad.abs().item(), 0)

    def test_module_replacement_substitutes_every_native_shared_delta(self):
        source = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        batch = self.collator([
            BalancedUnpairedDataset(source, seed=13)[0],
            BalancedUnpairedDataset(source, seed=13)[1],
        ])
        condition, target = batch["text_batch"], batch["image_batch"]
        model = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            use_modality_embeddings=False, shared_translation_layers=[1],
            shared_translation_mode="module_replacement",
        )
        from models import inject_tri_lora
        inject_tri_lora(
            model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 2, 0.0,
            delete_base_weights=True,
        )
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, TriLoRALinear):
                    module.shared_B.normal_(std=0.02)
            model.shared_translation_bridges["1"].replacement_gate.fill_(0.1)
        target_ids = target["input_ids"].masked_fill(
            target["eligible_mask"], self.tokenizer.mask_id
        )
        logits, attention = model.forward_shared_translation(
            condition["input_ids"], condition["attention_mask"], condition["position_ids"],
            condition["modality_ids"], condition["route_ids"], target_ids,
            target["attention_mask"], target["position_ids"], target["modality_ids"],
            target["route_ids"], return_bridge_attention=True,
            condition_kv_mask=condition["eligible_mask"],
        )
        self.assertEqual(logits.shape[:2], target_ids.shape)
        self.assertEqual(attention[1].shape[2], target_ids.size(1))
        logits.square().mean().backward()
        bridge = model.shared_translation_bridges["1"]
        self.assertIsNotNone(bridge.replacement_gate.grad)
        self.assertGreater(bridge.replacement_gate.grad.abs().item(), 0)
        layer_modules = [
            module for module in model.modules()
            if isinstance(module, TriLoRALinear) and module.layer_index == 1
        ]
        # The target's own shared delta is replaced, so condition-generated
        # native deltas carry gradients while the target private image branch
        # remains active in all four linear modules.
        self.assertEqual(len(layer_modules), 4)
        self.assertTrue(all(module.shared_B.grad is not None for module in layer_modules))
        self.assertTrue(all(module.image_B.grad is not None for module in layer_modules))

    def test_lora_freezes_base_and_trains_adapters(self):
        model = MultimodalMaskedTransformer(self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        names = inject_lora(model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 8, 0.0)
        self.assertEqual(len(names), 8)
        total, trainable = parameter_counts(model)
        self.assertGreater(trainable, 0)
        self.assertLess(trainable, total)

    def test_tri_lora_routes_private_branches(self):
        layer = TriLoRALinear(torch.nn.Linear(4, 4), rank=3, alpha=2, dropout=0, name="test")
        inputs = torch.randn(2, 3, 4)
        with lora_modality_context(torch.full((2,), TEXT_ROUTE_ID)):
            layer(inputs).sum().backward()
        self.assertIsNotNone(layer.shared_B.grad)
        self.assertIsNotNone(layer.text_B.grad)
        self.assertIsNone(layer.image_B.grad)
        layer.zero_grad(set_to_none=True)
        with lora_modality_context(torch.full((2,), IMAGE_ROUTE_ID)):
            layer(inputs).sum().backward()
        self.assertIsNotNone(layer.shared_B.grad)
        self.assertIsNone(layer.text_B.grad)
        self.assertIsNotNone(layer.image_B.grad)

    def test_block_masking_hides_rectangles_of_the_image_grid(self):
        torch.manual_seed(0)
        batch, height, width = 64, 16, 24
        cells = height * width
        length = cells + 3
        input_ids = torch.randint(170, 682, (batch, length))
        eligible = torch.zeros(batch, length, dtype=torch.bool)
        eligible[:, 2:2 + cells] = True
        modality = torch.full((batch, length), 2)
        corrupted, masked, _ = corrupt_batch(
            input_ids, eligible, modality, mask_token_id=1, objective="image",
            fixed_t=0.3, mask_block_grid=(height, width),
            mask_block_scale=(0.10, 0.25), mask_block_aspect=(0.75, 1.5),
        )
        self.assertFalse((masked & ~eligible).any())
        self.assertAlmostEqual(
            (masked.sum(1).float() / cells).mean().item(), 0.3, delta=0.06
        )
        grid = masked[:, 2:2 + cells].reshape(batch, height, width)
        # Every masked cell of a rectangle has a masked neighbour, so the mask
        # is made of regions rather than scattered codes.
        padded = torch.zeros(batch, height + 2, width + 2, dtype=torch.bool)
        padded[:, 1:-1, 1:-1] = grid
        neighbours = (
            padded[:, :-2, 1:-1] | padded[:, 2:, 1:-1]
            | padded[:, 1:-1, :-2] | padded[:, 1:-1, 2:]
        )
        self.assertGreater((grid & neighbours).sum().item() / grid.sum().item(), 0.95)
        self.assertTrue(torch.equal(corrupted[masked], torch.ones_like(corrupted[masked])))

    def test_window_masking_hides_contiguous_spans_within_eligible_tokens(self):
        torch.manual_seed(0)
        batch, length, caption = 128, 192, 92
        input_ids = torch.randint(7, 170, (batch, length))
        eligible = torch.zeros(batch, length, dtype=torch.bool)
        eligible[:, :caption] = True
        modality = torch.ones_like(input_ids)
        corrupted, masked, _ = corrupt_batch(
            input_ids, eligible, modality, mask_token_id=1, objective="text",
            fixed_t=0.3, mask_span_min=12, mask_span_max=24,
        )
        # Padding is never masked and the realized fraction tracks the target.
        self.assertFalse((masked & ~eligible).any())
        self.assertAlmostEqual(
            (masked.sum(1).float() / caption).mean().item(), 0.3, delta=0.05
        )
        # Masked positions come in runs, not scattered single tokens.
        run_starts = masked & ~torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool), masked[:, :-1]], dim=1
        )
        mean_run = (masked.sum(1).float() / run_starts.sum(1).clamp_min(1)).mean().item()
        self.assertGreater(mean_run, 8.0)
        self.assertTrue(torch.equal(corrupted[masked], torch.ones_like(corrupted[masked])))

    def test_data2vec_scratch_bootstrap_adds_global_and_variance_losses(self):
        torch.manual_seed(7)
        width, batch, length = 8, 4, 6
        model = MultimodalMaskedTransformer(
            32, 16, width, 2, 2, 2, 0.0,
            shared_jepa=True, data2vec_hidden=True,
        )
        student = {
            layer: torch.randn(batch, length, width, requires_grad=True)
            for layer in range(2)
        }
        teacher = {
            layer: torch.randn(batch, length, width)
            for layer in range(2)
        }
        prediction_mask = torch.zeros(batch, length, dtype=torch.bool)
        prediction_mask[:, 1:3] = True
        pool_mask = torch.zeros_like(prediction_mask)
        pool_mask[:, :5] = True
        loss, details = data2vec_hidden_loss(
            model, student, teacher, prediction_mask,
            top_k=2, beta=2.0, mode="average", pool_mask=pool_mask,
            token_weight=0.25, global_weight=1.0,
            variance_weight=0.1, variance_target=1.0,
        )
        self.assertGreater(details["token_loss"], 0)
        self.assertGreater(details["global_loss"], 0)
        self.assertGreaterEqual(details["variance_loss"], 0)
        self.assertTrue(-1 <= details["global_cosine"] <= 1)
        loss.backward()
        self.assertIsNotNone(student[1].grad)
        self.assertTrue(any(parameter.grad is not None for parameter in model.data2vec_head.parameters()))

    def test_data2vec_rejects_all_zero_component_weights(self):
        model = MultimodalMaskedTransformer(
            32, 16, 8, 2, 2, 2, 0.0,
            shared_jepa=True, data2vec_hidden=True,
        )
        hidden = {layer: torch.randn(2, 4, 8) for layer in range(2)}
        prediction_mask = torch.ones(2, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "non-zero loss weight"):
            data2vec_hidden_loss(
                model, hidden, hidden, prediction_mask,
                top_k=2, beta=2.0, token_weight=0.0,
            )

    def test_data2vec_pool_mask_is_validated(self):
        model = MultimodalMaskedTransformer(
            32, 16, 8, 2, 2, 2, 0.0,
            shared_jepa=True, data2vec_hidden=True,
        )
        hidden = {layer: torch.randn(2, 4, 8) for layer in range(2)}
        prediction_mask = torch.ones(2, 4, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "pool mask"):
            data2vec_hidden_loss(
                model, hidden, hidden, prediction_mask,
                top_k=2, beta=2.0, pool_mask=torch.ones(2, 3, dtype=torch.bool),
                global_weight=1.0,
            )

    def test_data2vec_forward_returns_pre_residual_ffn_targets(self):
        torch.manual_seed(3)
        model = MultimodalMaskedTransformer(
            32, 16, 8, 2, 2, 2, 0.0,
            shared_jepa=True, data2vec_hidden=True,
        ).eval()
        ids = torch.randint(0, 32, (2, 5))
        mask = torch.ones_like(ids, dtype=torch.bool)
        positions = torch.arange(5)[None].expand(2, -1)
        modalities = torch.ones_like(ids)
        routes = torch.zeros(2, dtype=torch.long)
        logits, residuals, ffn_outputs = model(
            ids, mask, positions, modalities, routes,
            return_data2vec_by_layer=True,
        )
        self.assertEqual(logits.shape, (2, 5, 32))
        self.assertEqual(set(residuals), {0, 1})
        self.assertEqual(set(ffn_outputs), {0, 1})
        # The FFN write is a target before residual addition, not an alias of
        # the complete block output.
        self.assertFalse(torch.allclose(residuals[1], ffn_outputs[1]))

    def test_data2vec_teacher_can_share_online_input_encoder(self):
        torch.manual_seed(5)
        student = MultimodalMaskedTransformer(
            32, 16, 8, 2, 2, 2, 0.0,
            shared_jepa=True, data2vec_hidden=True,
        ).eval()
        teacher = copy.deepcopy(student).eval()
        with torch.no_grad():
            teacher.token_embed.weight.add_(3.0)
        ids = torch.randint(0, 32, (2, 5))
        mask = torch.ones_like(ids, dtype=torch.bool)
        positions = torch.arange(5)[None].expand(2, -1)
        modalities = torch.ones_like(ids)
        routes = torch.zeros(2, dtype=torch.long)
        shared = student.input_embeddings(ids, positions, modalities)
        student_logits = student(ids, mask, positions, modalities, routes)
        shared_teacher_logits = teacher(
            ids, mask, positions, modalities, routes, input_embeddings=shared
        )
        private_teacher_logits = teacher(ids, mask, positions, modalities, routes)
        self.assertTrue(torch.allclose(student_logits, shared_teacher_logits))
        self.assertFalse(torch.allclose(student_logits, private_teacher_logits))

    def test_ema_can_skip_private_teacher_embedding_copies(self):
        torch.manual_seed(11)
        student = MultimodalMaskedTransformer(32, 16, 8, 2, 2, 2, 0.0)
        teacher = copy.deepcopy(student)
        old_embedding = teacher.token_embed.weight.detach().clone()
        old_backbone = teacher.blocks[0].norm1.weight.detach().clone()
        with torch.no_grad():
            student.token_embed.weight.add_(1.0)
            student.blocks[0].norm1.weight.add_(1.0)
        update_ema_teacher(teacher, student, 0.5, share_input_encoder=True)
        self.assertTrue(torch.equal(teacher.token_embed.weight, old_embedding))
        self.assertTrue(torch.allclose(
            teacher.blocks[0].norm1.weight, old_backbone + 0.5
        ))

    def test_data2vec_tristage_lr_shape(self):
        values = [data2vec_tristage_multiplier(step, 100, 0.05, 0.80) for step in range(101)]
        self.assertAlmostEqual(values[0], 0.2)
        self.assertAlmostEqual(values[4], 1.0)
        self.assertAlmostEqual(values[84], 1.0)
        self.assertGreater(values[85], values[99])
        self.assertEqual(values[100], 0.0)

    def test_bert_replacement_splits_selected_positions_80_10_10(self):
        torch.manual_seed(0)
        vocab = 40
        input_ids = torch.randint(7, vocab, (64, 96))
        eligible = torch.ones_like(input_ids, dtype=torch.bool)
        modality = torch.ones_like(input_ids)
        corrupted, masked, t = corrupt_batch(
            input_ids, eligible, modality, mask_token_id=1, objective="text",
            fixed_t=0.15, bert_replacement=True,
            random_token_ranges={1: (7, vocab)},
        )
        # The prediction mask still covers the full 15%, as in BERT.
        self.assertAlmostEqual(masked.float().mean().item(), 0.15, delta=0.01)
        self.assertTrue(torch.allclose(t, torch.full_like(t, 0.15)))
        selected = masked.sum().item()
        became_mask = (corrupted.eq(1) & masked).sum().item()
        unchanged = (corrupted.eq(input_ids) & masked).sum().item()
        self.assertAlmostEqual(became_mask / selected, 0.8, delta=0.03)
        # Unchanged covers the deliberate 10% plus the random draws that happen
        # to reproduce the original token (1 in 33 of the 10% randomized).
        self.assertAlmostEqual(unchanged / selected, 0.1, delta=0.03)
        # Positions outside the mask are never touched.
        self.assertTrue(torch.equal(corrupted[~masked], input_ids[~masked]))
        # Replacements stay inside the requested range.
        self.assertTrue(corrupted[masked].ge(1).all() and corrupted[masked].lt(vocab).all())

    def test_dense_private_keeps_base_as_shared_route(self):
        base = torch.nn.Linear(6, 6)
        reference = base.weight.detach().clone()
        layer = TriLoRALinear(
            base, rank=6, alpha=2, dropout=0, name="dense_private",
            delete_base_weights=False, shared_branch=False, private_rank=2,
        )
        self.assertFalse(layer.shared_branch_enabled)
        self.assertFalse(hasattr(layer, "shared_A"))
        self.assertEqual((layer.rank_text, layer.rank_image), (2, 2))
        self.assertTrue(torch.equal(layer.base.weight, reference))
        inputs = torch.randn(2, 3, 6)
        # Private adapters start at zero, so the layer must still be the dense
        # map it wrapped.
        with lora_modality_context(torch.full((2,), TEXT_ROUTE_ID)):
            output = layer(inputs)
        self.assertTrue(torch.allclose(output, F.linear(inputs, reference, layer.base.bias), atol=1e-6))
        with torch.no_grad():
            layer.text_B.normal_(0, 0.1)
        with shared_activation_context(None, 6) as recorder:
            with lora_modality_context(torch.full((2,), TEXT_ROUTE_ID)):
                layer(inputs)
        # HSIC, JEPA and the evaluators read the dense write as the shared route.
        self.assertTrue(torch.allclose(
            recorder.native_by_module["dense_private"],
            F.linear(inputs, reference, layer.base.bias),
            atol=1e-6,
        ))
        self.assertFalse(torch.allclose(
            recorder.native_private_by_module["dense_private"],
            torch.zeros_like(recorder.native_private_by_module["dense_private"]),
        ))

    def test_image_private_only_route_suppresses_shared_image_update(self):
        base = torch.nn.Linear(4, 4, bias=False)
        layer = TriLoRALinear(
            base, rank=3, alpha=2, dropout=0, name="asymmetric", delete_base_weights=True
        )
        with torch.no_grad():
            layer.shared_B.fill_(1.0)
            layer.image_B.fill_(1.0)
        inputs = torch.randn(2, 3, 4)
        routes = torch.full((2, 3), IMAGE_PRIVATE_ONLY_ROUTE_ID)
        with lora_modality_context(routes):
            output = layer(inputs)
        expected = layer.shared_bias + layer._delta(inputs, "image")
        self.assertTrue(torch.allclose(output, expected))
        output.sum().backward()
        self.assertTrue(
            layer.shared_B.grad is None or torch.count_nonzero(layer.shared_B.grad).item() == 0
        )
        self.assertIsNotNone(layer.image_B.grad)

    def test_shared_only_stage_deletes_base_and_delays_private_branches(self):
        model = MultimodalMaskedTransformer(self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        from models import inject_tri_lora
        inject_tri_lora(
            model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 2, 0.0,
            delete_base_weights=True,
        )
        args = Namespace(
            lr=3e-4, shared_lora_lr=1.5e-4, private_lora_lr=3e-4,
            embedding_lr=3e-4, modality_discriminator_lr=3e-4,
        )
        groups = optimizer_groups(model, args)
        private_parameters = [
            parameter
            for _, module in model.named_modules() if isinstance(module, TriLoRALinear)
            for parameter in (module.text_A, module.text_B, module.image_A, module.image_B)
        ]
        optimized_ids = {id(parameter) for group in groups for parameter in group["params"]}
        self.assertTrue(all(id(parameter) in optimized_ids for parameter in private_parameters))

        count = set_private_lora_trainable(model, False)
        tri_modules = [module for module in model.modules() if isinstance(module, TriLoRALinear)]
        self.assertEqual(count, 4 * len(tri_modules))
        self.assertTrue(all(module.base.weight is None for module in tri_modules))
        self.assertTrue(all(module.base.bias is None for module in tri_modules))
        self.assertTrue(all(module.shared_bias.requires_grad for module in tri_modules))
        self.assertTrue(all(module.shared_A.requires_grad and module.shared_B.requires_grad for module in tri_modules))
        self.assertTrue(all(not parameter.requires_grad for parameter in private_parameters))

        set_private_lora_trainable(model, True)
        self.assertTrue(all(parameter.requires_grad for parameter in private_parameters))

    def test_no_base_task_loss_reaches_shared_and_private_adapters_without_dann(self):
        dataset = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        batch = self.collator([
            dataset.get_modality(0, "text"), dataset.get_modality(1, "text")
        ])
        model = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            use_modality_embeddings=False,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        from models import inject_tri_lora
        inject_tri_lora(
            model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 2, 0.0,
            delete_base_weights=True,
        )
        for module in model.modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.LayerNorm)):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        for parameter in model.head.parameters():
            parameter.requires_grad_(True)

        logits = model(
            batch["input_ids"], batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"],
        )
        eligible = batch["eligible_mask"]
        torch.nn.functional.cross_entropy(
            logits[eligible], batch["input_ids"][eligible]
        ).backward()
        tri_modules = [
            module for module in model.modules() if isinstance(module, TriLoRALinear)
        ]
        self.assertTrue(any(
            module.shared_B.grad is not None and module.shared_B.grad.abs().sum().item() > 0
            for module in tri_modules
        ))
        self.assertTrue(any(
            module.text_B.grad is not None and module.text_B.grad.abs().sum().item() > 0
            for module in tri_modules
        ))
        self.assertTrue(all(module.image_B.grad is None for module in tri_modules))

    def test_shared_jepa_predicts_clean_masked_token_latents_without_private_gradients(self):
        dataset = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        batch = self.collator([
            dataset.get_modality(0, "text"), dataset.get_modality(1, "text")
        ])
        model = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            use_modality_embeddings=False,
            shared_jepa=True,
            shared_jepa_predictor_hidden_multiplier=2,
        )
        from models import inject_tri_lora
        inject_tri_lora(
            model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 2, 0.0,
            delete_base_weights=True,
        )
        tri_modules = [module for module in model.modules() if isinstance(module, TriLoRALinear)]
        with torch.no_grad():
            for module in tri_modules:
                module.shared_B.normal_(std=0.02)
        masked = batch["eligible_mask"].clone()
        corrupted = batch["input_ids"].clone()
        corrupted[masked] = self.tokenizer.mask_id
        masked_output = model(
            corrupted, batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"], return_shared=True,
            return_shared_tokens_by_layer=True,
        )
        with torch.no_grad():
            clean_output = model(
                batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                batch["modality_ids"], batch["route_ids"], return_shared=True,
                return_shared_tokens_by_layer=True,
            )
        loss, layer_losses, layer_cosines = shared_latent_jepa_loss(
            model, masked_output[3], clean_output[3], masked
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(layer_losses), {0, 1})
        self.assertEqual(set(layer_cosines), {0, 1})
        loss.backward()
        self.assertTrue(any(
            module.shared_B.grad is not None and module.shared_B.grad.abs().sum().item() > 0
            for module in tri_modules
        ))
        self.assertTrue(all(module.text_B.grad is None for module in tri_modules))
        self.assertTrue(all(module.image_B.grad is None for module in tri_modules))
        self.assertIsNone(model.token_embed.weight.grad)
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum().item() > 0
            for parameter in model.shared_jepa_predictors.parameters()
        ))

        normalized_loss, selected_losses, selected_cosines = shared_latent_jepa_loss(
            model, masked_output[3], clean_output[3], masked,
            layers=[1], loss_type="normalized_mse",
        )
        scaled_target_loss, _, _ = shared_latent_jepa_loss(
            model, masked_output[3],
            {layer: value * 10 for layer, value in clean_output[3].items()},
            masked, layers=[1], loss_type="normalized_mse",
        )
        self.assertEqual(set(selected_losses), {1})
        self.assertEqual(set(selected_cosines), {1})
        self.assertTrue(torch.allclose(normalized_loss, scaled_target_loss, atol=1e-5))

    def test_objective_gradient_diagnostics_do_not_populate_parameter_grad(self):
        dataset = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        batch = self.collator([
            dataset.get_modality(0, "text"), dataset.get_modality(1, "text")
        ])
        model = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            use_modality_embeddings=False, shared_jepa=True,
        )
        from models import inject_tri_lora
        inject_tri_lora(
            model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 2, 0.0,
            delete_base_weights=True,
        )
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, TriLoRALinear):
                    module.shared_B.normal_(std=0.02)
        masked = batch["eligible_mask"].clone()
        corrupted = batch["input_ids"].masked_fill(masked, self.tokenizer.mask_id)
        output = model(
            corrupted, batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"], return_shared=True,
            return_shared_tokens_by_layer=True,
        )
        with torch.no_grad():
            clean = model(
                batch["input_ids"], batch["attention_mask"], batch["position_ids"],
                batch["modality_ids"], batch["route_ids"], return_shared=True,
                return_shared_tokens_by_layer=True,
            )
        diffusion = torch.nn.functional.cross_entropy(
            output[0].transpose(1, 2), batch["input_ids"], reduction="none"
        )[masked].mean()
        jepa, _, _ = shared_latent_jepa_loss(model, output[3], clean[3], masked)
        sigreg_proxy = output[1].float().square().mean()
        metrics = objective_shared_gradient_metrics(
            model, {"diffusion": diffusion, "jepa": 0.1 * jepa, "sigreg": 0.01 * sigreg_proxy}
        )
        self.assertIn("layer_00/jepa_to_diffusion_norm_ratio", metrics)
        self.assertIn("layer_01/diffusion_sigreg_cosine", metrics)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_paired_tokens_use_their_own_private_branches(self):
        dataset = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "paired")
        batch = self.collator([dataset[0], dataset[1]])
        routes = batch["route_ids"]
        self.assertEqual(routes.shape, batch["modality_ids"].shape)
        self.assertTrue(routes[batch["modality_ids"].eq(1)].eq(TEXT_ROUTE_ID).all())
        self.assertTrue(routes[batch["modality_ids"].eq(2)].eq(IMAGE_ROUTE_ID).all())
        self.assertTrue(routes[~batch["attention_mask"]].eq(-1).all())

        layer = TriLoRALinear(torch.nn.Linear(4, 4), rank=3, alpha=2, dropout=0, name="mixed")
        inputs = torch.randn(2, routes.size(1), 4)
        with lora_modality_context(routes):
            layer(inputs).sum().backward()
        self.assertIsNotNone(layer.shared_B.grad)
        self.assertIsNotNone(layer.text_B.grad)
        self.assertIsNotNone(layer.image_B.grad)

    def test_paired_both_is_bidirectional(self):
        dataset = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "paired")
        batch = self.collator([dataset[0], dataset[1]])
        specifications = modality_sub_batches(batch, Namespace(objective="both"))
        self.assertEqual(
            [(name, objective, route) for name, _, objective, route in specifications],
            [("text", "text", TEXT_ROUTE_ID), ("image", "image", IMAGE_ROUTE_ID)],
        )

    def test_role_routed_paired_batch_is_split_and_targets_are_private_only(self):
        dataset = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "paired")
        batch = self.collator([dataset[0], dataset[1]])
        specifications = modality_sub_batches(
            batch, Namespace(objective="both", asymmetric_condition_target=True)
        )
        text_batch = specifications[0][1]
        image_batch = specifications[1][1]
        self.assertEqual(text_batch["input_ids"].size(0), 1)
        self.assertEqual(image_batch["input_ids"].size(0), 1)
        self.assertTrue(
            text_batch["route_ids"][text_batch["modality_ids"].eq(1)]
            .eq(TEXT_PRIVATE_ONLY_ROUTE_ID).all()
        )
        self.assertTrue(
            text_batch["route_ids"][text_batch["modality_ids"].eq(2)]
            .eq(IMAGE_ROUTE_ID).all()
        )
        self.assertTrue(
            image_batch["route_ids"][image_batch["modality_ids"].eq(2)]
            .eq(IMAGE_PRIVATE_ONLY_ROUTE_ID).all()
        )
        self.assertTrue(
            image_batch["route_ids"][image_batch["modality_ids"].eq(1)]
            .eq(TEXT_ROUTE_ID).all()
        )

    def test_paired_alignment_controls_keep_target_and_mask_fixed(self):
        dataset = ClevrMultimodalDataset(
            self.root, self.root / "tokens.pt", "paired",
            pair_manifest=self.root / "pairs.jsonl", caption_field="caption_human",
        )
        batch = self.collator([dataset[0], dataset[1], dataset[2]])
        matched = _conditioned_batch(batch, "image", "matched")
        shuffled = _conditioned_batch(batch, "image", "shuffled")
        null = _conditioned_batch(batch, "image", "null")
        specs = _target_mask_spec(matched, "image", 0.75, seed=7)

        target_rows = []
        masks = []
        for conditioned in (matched, shuffled, null):
            _, masked = _apply_target_mask(conditioned, "image", specs, self.tokenizer.mask_id)
            masks.append([masked[row][conditioned["modality_ids"][row].eq(2)].tolist() for row in range(3)])
            target_rows.append([
                conditioned["input_ids"][row][conditioned["modality_ids"][row].eq(2)].tolist()
                for row in range(3)
            ])
        self.assertEqual(target_rows[0], target_rows[1])
        self.assertEqual(target_rows[0], target_rows[2])
        self.assertEqual(masks[0], masks[1])
        self.assertEqual(masks[0], masks[2])
        self.assertFalse(null["modality_ids"].eq(1).any())
        self.assertNotEqual(
            matched["input_ids"][0][matched["modality_ids"][0].eq(1)].tolist(),
            shuffled["input_ids"][0][shuffled["modality_ids"][0].eq(1)].tolist(),
        )

    def test_forced_full_mask_corruption(self):
        dataset = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "paired")
        batch = self.collator([dataset[0], dataset[1]])
        _, masked, t = corrupt_batch(
            batch["input_ids"], batch["eligible_mask"], batch["modality_ids"],
            self.tokenizer.mask_id, objective="image", full_mask_probability=1.0,
        )
        selected = batch["eligible_mask"] & batch["modality_ids"].eq(2)
        self.assertTrue(torch.equal(masked, selected))
        self.assertTrue(t.eq(1).all())

    def test_paired_selection_and_adversary_can_be_dropped_on_init(self):
        metrics = {
            "val/paired/text_to_image/t1/matched_loss": 2.0,
            "val/paired/image_to_text/t1/matched_loss": 4.0,
        }
        self.assertEqual(paired_t1_selection_loss(metrics), 3.0)
        self.assertEqual(paired_t1_selection_loss(metrics, "text_to_image"), 2.0)
        source = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            modality_adversarial=True, modality_discriminator_hidden=8,
        )
        target = MultimodalMaskedTransformer(self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0)
        load_initial_model_weights(target, {"model": source.state_dict()})

    def test_gradient_reversal_and_shared_adversary(self):
        value = torch.ones(3, requires_grad=True)
        gradient_reverse(value, 0.5).sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.full((3,), -0.5)))

        dataset = ClevrMultimodalDataset(self.root, self.root / "tokens.pt", "unpaired")
        batch = self.collator([dataset.get_modality(0, "text"), dataset.get_modality(1, "text")])
        model = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            modality_adversarial=True, modality_discriminator_hidden=8,
        )
        from models import inject_tri_lora
        inject_tri_lora(model, ["qkv", "out_proj", "mlp.0", "mlp.3"], 4, 2, 0.0)
        logits, shared = model(
            batch["input_ids"], batch["attention_mask"], batch["position_ids"],
            batch["modality_ids"], batch["route_ids"], return_shared=True,
        )
        self.assertEqual(tuple(shared.shape), (2, 32))
        modality_logits = model.modality_logits(shared)
        self.assertEqual(tuple(modality_logits.shape), (2, 2))
        self.assertEqual(logits.shape[:2], batch["input_ids"].shape)
        torch.nn.functional.cross_entropy(
            modality_logits, torch.full((2,), TEXT_ROUTE_ID)
        ).backward()
        tri_modules = [
            (name, module) for name, module in model.named_modules()
            if isinstance(module, TriLoRALinear)
        ]
        self.assertEqual(len(tri_modules), 8)
        # In a strict no-base network all B matrices start at zero.  A later
        # layer can therefore have zero input on the very first backward pass;
        # require DANN to reach shared branches, not every layer immediately.
        self.assertTrue(any(
            module.shared_B.grad is not None and module.shared_B.grad.abs().sum().item()
            for _, module in tri_modules
        ))
        self.assertTrue(all(module.text_B.grad is None for _, module in tri_modules))
        self.assertTrue(all(module.image_B.grad is None for _, module in tri_modules))

    def test_l2_normalized_dann_input_has_unit_norm_and_reversed_gradient(self):
        model = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            modality_adversarial=True,
            modality_discriminator_hidden=8,
            modality_adversarial_representation_normalization="l2",
        )
        shared = torch.randn(3, 32, requires_grad=True)
        normalized = model.modality_discriminator_input(shared)
        self.assertTrue(torch.allclose(normalized.norm(dim=-1), torch.ones(3), atol=1e-5))
        torch.nn.functional.cross_entropy(
            model.modality_logits(shared, reversal_coefficient=1.0), torch.tensor([0, 1, 0])
        ).backward()
        self.assertIsNotNone(shared.grad)

    def test_modality_embedding_can_be_disabled(self):
        """Without the optional table, modality IDs cannot alter the input state."""
        model = MultimodalMaskedTransformer(
            self.collator.vocab_size, 16, 32, 2, 4, 2, 0.0,
            use_modality_embeddings=False,
        ).eval()
        ids = torch.tensor([[1, 2, 3, 4]])
        positions = torch.tensor([[0, 1, 2, 3]])
        attention = torch.ones_like(ids, dtype=torch.bool)
        text_ids = torch.ones_like(ids)
        image_ids = torch.full_like(ids, 2)
        with torch.no_grad():
            text_logits = model(ids, attention, positions, text_ids)
            image_logits = model(ids, attention, positions, image_ids)
        self.assertIsNone(model.modality_embed)
        self.assertTrue(torch.equal(text_logits, image_logits))


if __name__ == "__main__":
    unittest.main()
