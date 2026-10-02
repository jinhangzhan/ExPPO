"""CPU tests of prompt collection and both reward-manager implementations."""

import ast
import asyncio
from collections import defaultdict
from contextlib import contextmanager
import importlib.util
import inspect
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / "third_party/verl/verl"
sys.path.insert(0, str(VERL.parent))
try:
    from verl import DataProto
except ImportError:
    DataProto = None

spec = importlib.util.spec_from_file_location("dynamic_sampling", VERL / "trainer/ppo/dynamic_sampling.py")
dynamic_sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamic_sampling)
DynamicGroupCollector = dynamic_sampling.DynamicGroupCollector


def batch_for(groups):
    uids, acc = zip(*[(uid, score) for uid, scores in groups.items() for score in scores])
    size = len(acc)
    return DataProto.from_dict(
        tensors={"token_level_scores": torch.tensor(acc)[:, None].float(),
                 "responses": torch.arange(size)[:, None]},
        non_tensors={"uid": np.asarray(uids, dtype=object), "acc": np.asarray(acc),
                     "row_id": np.asarray([f"{uid}/{row}" for row, uid in enumerate(uids)])},
        meta_info={"temperature": 1.0, "global_steps": 1,
                   "global_token_num": list(range(size))},
    )


@unittest.skipIf(DataProto is None, "Install training requirements for DataProto tests")
class DynamicSamplingTests(unittest.TestCase):
    def test_accumulates_complete_informative_groups(self):
        collector = DynamicGroupCollector(2, 2, max_num_gen_batches=3)
        first = batch_for({"uniform-good": [1, 1], "a": [1, 0], "uniform-bad": [0, 0]})
        self.assertIsNone(collector.add(first, ["acc"]))
        second = batch_for({"b": [0, 1], "discard-excess": [1, 0]})
        result = collector.add(second, ["acc"])
        self.assertEqual(result.non_tensor_batch["uid"].tolist(), ["a", "a", "b", "b"])
        np.testing.assert_array_equal(result.non_tensor_batch["acc"], [1, 0, 0, 1])
        torch.testing.assert_close(result.batch["token_level_scores"].flatten(), torch.tensor([1., 0., 0., 1.]))
        self.assertEqual(collector.metrics()["dapo/generation_batches"], 2)
        self.assertEqual(collector.metrics()["dapo/generated_prompts"], 5)
        self.assertNotIn("global_token_num", result.meta_info)
        collector.reset()
        self.assertEqual(collector.accepted_prompts, 0)
        self.assertEqual(collector.parts, [])

    def test_raw_correctness_filters_even_when_penalties_differ(self):
        collector = DynamicGroupCollector(1, 2)
        batch = batch_for({"uniform": [1, 1], "mixed": [1, 0]})
        batch.batch["token_level_scores"] = torch.tensor([[1.], [.2], [1.], [-.5]])
        result = collector.add(batch, ["acc"])
        self.assertEqual(result.non_tensor_batch["uid"].tolist(), ["mixed", "mixed"])

    def test_reordered_inputs_keep_diagnostics_and_full_groups_aligned(self):
        collector = DynamicGroupCollector(1, 2)
        batch = batch_for({"a": [0, 1], "b": [1, 0]})
        batch.reorder(torch.tensor([0, 2, 1, 3]))
        result = collector.add(batch, ["acc"])
        self.assertEqual(result.non_tensor_batch["row_id"].tolist(), ["a/0", "a/1"])
        result.reorder(torch.tensor([1, 0]))  # Training balances after collection.
        self.assertEqual(result.non_tensor_batch["row_id"].tolist(), ["a/1", "a/0"])
        np.testing.assert_array_equal(result.non_tensor_batch["acc"], [1, 0])
        self.assertEqual(result.batch["responses"].flatten().tolist(), [1, 0])

    def test_fails_at_candidate_limit_and_rejects_incomplete_groups(self):
        collector = DynamicGroupCollector(2, 2, max_num_gen_batches=1)
        with self.assertRaisesRegex(RuntimeError, "1/2 informative"):
            collector.add(batch_for({"a": [0, 1]}), ["acc"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DynamicGroupCollector(1, 2).add(batch_for({"a": [1]}), ["acc"])

    def test_metric_validation_and_unlimited_collection(self):
        collector = DynamicGroupCollector(1, 2, max_num_gen_batches=0)
        for i in range(40):
            self.assertIsNone(collector.add(batch_for({str(i): [0, 0]}), ["acc"]))
        self.assertEqual(len(collector.add(batch_for({"mixed": [0, 1]}), ["acc"])), 2)
        batch = batch_for({"a": [0, 1]})
        batch.non_tensor_batch["acc"] = np.array([np.nan, 1])
        with self.assertRaisesRegex(ValueError, "finite"):
            DynamicGroupCollector(1, 2).add(batch, ["acc"])
        with self.assertRaisesRegex(ValueError, "absent"):
            DynamicGroupCollector(1, 2, metric="missing").add(batch, ["acc"])

    def test_sequence_reward_metrics(self):
        batch = batch_for({"a": [1, 1], "b": [0, 1]})
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        for metric in ("seq_reward", "seq_final_reward"):
            result = DynamicGroupCollector(1, 2, metric=metric).add(batch, ["acc"])
            self.assertEqual(result.non_tensor_batch["uid"].tolist(), ["b", "b"])


def load_reward_class(relative_path, *, async_manager=False):
    """Execute the actual class without importing Ray/GPU reward-loop workers."""
    path = VERL / relative_path
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DAPORewardManager")
    node.decorator_list = []
    namespace = {"torch": torch, "DataProto": DataProto, "defaultdict": defaultdict, "inspect": inspect,
                 "default_compute_score": None,
                 "AbstractRewardManager": type("AbstractRewardManager", (), {
                     "_extract_reward_from_rm_scores": lambda *_: None}),
                 "RewardManagerBase": type("RewardManagerBase", (), {
                     "__init__": lambda self, config, tokenizer: (
                         setattr(self, "tokenizer", tokenizer), setattr(self, "loop", asyncio.get_running_loop()), None
                     )[-1]})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["DAPORewardManager"]


@unittest.skipIf(DataProto is None, "Install training requirements for DataProto tests")
class DAPORewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(enable=True, len=2, penalty_factor=1.0, log=True)
        self.tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: "response", eos_token="<eos>")

    def reward_batch(self):
        return DataProto.from_dict(
            tensors={"prompts": torch.ones(3, 1, dtype=torch.long),
                     "responses": torch.ones(3, 6, dtype=torch.long),
                     "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0],
                                                     [1, 1, 1, 1, 1, 1, 0],
                                                     [1, 1, 1, 1, 1, 1, 1]])},
            non_tensors={"data_source": np.array(["math"] * 3, dtype=object),
                         "reward_model": np.array([{"ground_truth": "x"} for _ in range(3)], dtype=object),
                         "extra_info": np.array([{} for _ in range(3)], dtype=object)},
        )

    def test_legacy_penalty_and_raw_correctness(self):
        manager_cls = load_reward_class("workers/reward_manager/dapo.py")
        manager = manager_cls(self.tokenizer, 0, compute_score=lambda **kwargs: 1.0,
                              max_resp_len=6, overlong_buffer_cfg=self.cfg)
        result = manager(self.reward_batch(), return_dict=True)
        torch.testing.assert_close(result["reward_tensor"].sum(-1), torch.tensor([1., .5, 0.]))
        self.assertEqual(result["reward_extra_info"]["acc"], [1., 1., 1.])
        manager.overlong_buffer_cfg = None
        torch.testing.assert_close(manager(self.reward_batch()).sum(-1), torch.ones(3))

    def test_async_penalty_matches_legacy(self):
        async def run():
            manager_cls = load_reward_class("experimental/reward_loop/reward_manager/dapo.py")
            config = SimpleNamespace(reward_model={"reward_kwargs": {
                "max_resp_len": 6, "overlong_buffer_cfg": self.cfg}})
            manager = manager_cls(config, self.tokenizer, compute_score=lambda **kwargs: 1.0)
            batch = self.reward_batch()
            results = [await manager.run_single(batch[i:i + 1]) for i in range(3)]
            self.assertEqual([float(row["reward_score"]) for row in results], [1., .5, 0.])
            self.assertEqual([row["reward_extra_info"]["acc"] for row in results], [1., 1., 1.])
        asyncio.run(run())


@unittest.skipIf(DataProto is None, "Install training requirements for DataProto tests")
class DAPOTrainerFlowTests(unittest.TestCase):
    def test_generation_skips_preserve_update_count_and_reward_alignment(self):
        """Run the actual fit method with CPU worker stand-ins for two updates."""
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        previous_directory = Path.cwd()
        try:
            os.chdir(ROOT)
            with initialize_config_dir(config_dir=str(ROOT / "configs/training"), version_base=None):
                config = compose(config_name="paper", overrides=[
                    "algorithm.filter_groups.enable=true", "algorithm.filter_groups.metric=acc",
                    "data.train_batch_size=2", "actor_rollout_ref.rollout.n=2",
                    "trainer.total_epochs=1", "trainer.total_training_steps=2", "trainer.save_freq=1"])
        finally:
            os.chdir(previous_directory)

        records = defaultdict(list)

        @contextmanager
        def timer(name, timing, **kwargs):
            yield
            timing[name] = timing.get(name, 0) + 1

        class Logger:
            def __init__(self, **kwargs):
                pass
            def log(self, data, step):
                records["logged"].append((step, dict(data)))

        def generate(batch):
            # One uniform and one mixed group in every candidate batch.
            scores = torch.tensor([0, 0, 0, 1])
            records["generated"].append(trainer.global_steps)
            return DataProto.from_dict(
                tensors={"responses": scores[:, None],
                         "attention_mask": torch.ones(4, 2, dtype=torch.long)},
                meta_info={"timing": {"worker_generation": 1.0}},
            )

        def balance(batch, metrics):
            # If rewards are kept outside DataProto, this reversal exposes it.
            batch.reorder(torch.arange(len(batch) - 1, -1, -1))
            records["balanced"].append(len(batch))

        def old_log_prob(batch):
            return DataProto.from_dict(tensors={
                "old_log_probs": torch.zeros(len(batch), 1), "entropys": torch.ones(len(batch), 1)}), 0.0

        def update(batch):
            torch.testing.assert_close(batch.batch["token_level_rewards"], batch.batch["responses"].float())
            np.testing.assert_array_equal(batch.non_tensor_batch["acc"], batch.batch["responses"].flatten().numpy())
            records["updates"].append(trainer.global_steps)
            return SimpleNamespace(meta_info={"metrics": {}})

        class Loader(list):
            sampler = None

        candidates = Loader([{
            "prompts": torch.ones(2, 1, dtype=torch.long),
            "input_ids": torch.ones(2, 1, dtype=torch.long),
            "attention_mask": torch.ones(2, 1, dtype=torch.long),
            "multi_modal_inputs": np.array([{}, {}], dtype=object),
        } for _ in range(4)])
        trainer = SimpleNamespace(
            config=config, total_training_steps=2, train_dataloader=candidates, train_dataset=None,
            actor_rollout_wg=SimpleNamespace(), async_rollout_mode=True,
            async_rollout_manager=SimpleNamespace(generate_sequences=generate),
            checkpoint_manager=SimpleNamespace(
                update_weights=lambda: records["weight_updates"].append(trainer.global_steps),
                sleep_replicas=lambda: records["sleep"].append(trainer.global_steps)),
            resource_pool_manager=SimpleNamespace(get_n_gpus=lambda: 1),
            use_rm=False, use_reward_loop=False, use_critic=False, use_reference_policy=False,
            repo_r_enabled=False, cue_grpo_enabled=False, reward_fn=None,
            _load_checkpoint=lambda: None, _start_profiling=lambda flag: None,
            _stop_profiling=lambda flag: None, _get_gen_batch=lambda batch: batch,
            _balance_batch=balance, _compute_old_log_prob=old_log_prob, _update_actor=update,
            _compute_reward_legacy=lambda batch, **kwargs: (
                batch.batch["responses"].float(), {"acc": batch.batch["responses"].flatten().tolist()}),
            _save_checkpoint=lambda: records["saved"].append(trainer.global_steps),
        )
        source = VERL / "trainer/ppo/ray_trainer.py"
        tree = ast.parse(source.read_text())
        trainer_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
        method = next(node for node in trainer_cls.body if isinstance(node, ast.FunctionDef) and node.name == "fit")
        method.body = [node for node in method.body if not isinstance(node, ast.ImportFrom)]
        namespace = {"OmegaConf": OmegaConf, "Tracking": Logger, "os": os, "torch": torch, "np": np,
                     "uuid": uuid, "pprint": lambda value: None, "DynamicGroupCollector": DynamicGroupCollector,
                     "DataProto": DataProto, "marked_timer": timer,
                     "tqdm": lambda **kwargs: SimpleNamespace(update=lambda n: None, close=lambda: None),
                     "AdvantageEstimator": SimpleNamespace(GRPO="grpo", REMAX="remax"),
                     "compute_response_mask": lambda batch: batch.batch["attention_mask"][:, -1:],
                     "agg_loss": lambda **kwargs: torch.tensor(1.),
                     "compute_advantage": lambda batch, **kwargs: batch,
                     "reduce_metrics": lambda value: value,
                     "should_save_ckpt_esi": lambda **kwargs: False,
                     "compute_data_metrics": lambda **kwargs: {},
                     "compute_timing_metrics": lambda **kwargs: {},
                     "compute_throughout_metrics": lambda **kwargs: {},
                     "compute_variance_proxy_metrics": lambda **kwargs: {},
                     "AbstractCurriculumSampler": type("UnusedSampler", (), {}),
                     "trim_cpu_memory": lambda: None}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        namespace["fit"](trainer)
        self.assertEqual(records["generated"], [1, 1, 2, 2])
        self.assertEqual(records["updates"], [1, 2])
        self.assertEqual(records["saved"], [1, 2])
        self.assertEqual(records["sleep"], [1, 2])
        self.assertEqual(records["weight_updates"], [0, 1, 2])
        self.assertEqual(records["balanced"], [4, 4])
        self.assertEqual([row[1]["dapo/generation_batches"] for row in records["logged"]], [2, 2])

        # Direct trainer resumption is guarded even without train.sh's marker check.
        trainer._load_checkpoint = lambda: setattr(trainer, "global_steps", 2)
        namespace["fit"](trainer)
        self.assertEqual(records["generated"], [1, 1, 2, 2])
        self.assertEqual(records["updates"], [1, 2])

        # Candidate consumption cannot silently shorten the requested run.
        trainer._load_checkpoint = lambda: None
        trainer.total_training_steps = 3
        trainer.train_dataloader = Loader(candidates[:2])
        with self.assertRaisesRegex(RuntimeError, "1/3 optimizer updates"):
            namespace["fit"](trainer)


if __name__ == "__main__":
    unittest.main()
