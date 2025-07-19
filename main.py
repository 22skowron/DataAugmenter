from pathlib import Path

from data_augmenter import ParaphrasingAugmenter
from utils.timer import Timer
from utils.logger import logger


augmenter = ParaphrasingAugmenter(Path("M:/HuggingFaceModels/Phi-4-mini-instruct"))

if __name__ == '__main__':
    try:
        with Timer() as t:
                augmenter.run(
                    input_file=Path("input/sample_input_chunks.jsonl"),
                    output_file=Path("output/paraphrased.jsonl"),
                    prompt_type="basic",
                    num_augmentations=2,
                    batch_size=32,
                    start_line_idx=1,
                    temperature=0.7,
                    len_factor=1.4
                )
    finally:
        logger.info(f"Script execution finished. Processing time: {t.get_time_str()}")
