"""알려진 PPO epoch 결함 계열의 설정 전달 개발 변형. 모델에는 이름을 보내지 않는다."""
from __future__ import annotations

CONFIG = {"num_learning_epochs": 5, "num_mini_batches": 4,
          "learning_rate": 0.001, "schedule": "fixed"}
CONFIGURATIONS = ((5, 4), (3, 2), (2, 3))
SOURCES = {
    "healthy": '''def build_algorithm_kwargs(config):
    options = dict(config)
    return options
''',
    "omitted_argument": '''def build_algorithm_kwargs(config):
    options = {"num_mini_batches": config["num_mini_batches"],
               "learning_rate": config["learning_rate"],
               "schedule": config["schedule"]}
    return options
''',
    "literal_override": '''def build_algorithm_kwargs(config):
    options = dict(config)
    options["num_learning_epochs"] = 1
    return options
''',
    "late_override": '''def build_algorithm_kwargs(config):
    options = dict(config)
    options["num_learning_epochs"] = config["num_learning_epochs"]
    options["num_learning_epochs"] = 1
    return options
''',
}


def configurations() -> list[dict]:
    values = [{**CONFIG, "num_learning_epochs": epochs, "num_mini_batches": batches}
              for epochs, batches in CONFIGURATIONS]
    values[-1].update(clip_param=0.17, gamma=0.97)
    return values


def messages(context: dict) -> list[dict]:
    import json
    return [{"role": "system", "content": (
        "You investigate an optimizer configuration adapter used by an actual PPO implementation. "
        "Use only the provided source, configured parameters and measured CPU update facts. "
        "If the adapter already preserves the configuration and the measured updates conform, choose hold without changing it. "
        "Otherwise propose a corrected replacement of the single build_algorithm_kwargs(config) function. "
        "Preserve all configured algorithm parameters. Do not hardcode epoch or minibatch counts. "
        "The function must be pure: only local dict construction, config reads, assignments, and return; "
        "no imports, loops, classes, nested functions, external calls or file/process access. "
        "Return one JSON object with exact keys action, source_sha256, replacement_source, evidence_ids, rationale. "
        "action is patch or hold; replacement_source is the entire replacement source for patch, null for hold. "
        "Copy source_sha256 from the context and cite at least one exact existing fact ID. "
        "A proposal does not authorize execution or imply recovery."
    )}, {"role": "user", "content": json.dumps(context, ensure_ascii=False, allow_nan=False)}]
