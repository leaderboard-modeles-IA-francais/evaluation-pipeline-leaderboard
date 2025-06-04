import os
from pathlib import Path
from clearml import Task
from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.utils import EnvConfig

def main():
    output_dir=os.environ.get("OUTPUT_DIR")

    tasks_path=Path(f"french_evals.py")

    task = Task.init(project_name = "LLM Leaderboard FR", task_name = "eval_model")

    # Default task parameters, overwritten with task.connect(parameters) by the task actual parameters
    parameters = {
        'model': 'meta-llama/Llama-3.2-3B-Instruct',
        'dtype': 'bfloat16',
        'gpu_memory_utilization': 0.5,
        'nb_gpus_per_node': 4,
        'nb_nodes': 1,
        'enforce_eager': True,
        'tasks': 'community|bac-fr|0|0,community|ifeval-fr|0|0,community|gpqa-fr|0|0',
        'max_model_length': None,
        'use_chat_template': True,
    }

    task.connect(parameters)

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        custom_tasks_directory=tasks_path,
        use_chat_template=parameters['use_chat_template'],
    )

    model_config = VLLMModelConfig(
        pretrained=parameters['model'],
        dtype=parameters['dtype'],
        gpu_memory_utilization=parameters['gpu_memory_utilization'],
        tensor_parallel_size=parameters['nb_nodes'] * parameters['nb_gpus_per_node'],
        enforce_eager=parameters['enforce_eager'],
        max_model_length=parameters['max_model_length'],
        use_chat_template=parameters['use_chat_template'],
    )

    pipeline = Pipeline(
        tasks=parameters["tasks"],
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.save_and_push_results()
    pipeline.show_results()

    final = pipeline.get_results()
    logger = task.get_logger()

    for task_name, metrics in final['results'].items():
        for k, v in metrics.items():
            logger.report_single_value(f"{task_name} | {k}", v)

            task.upload_artifact(name='results', artifact_object=output_dir)

if __name__ == "__main__":
    main()
