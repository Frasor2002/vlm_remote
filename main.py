import argparse
from evaluate import test_mnist, test_wb, test_chc

# Short name -> full model path
AVAILABLE_MODELS = {
  "qwen_vl": "Qwen/Qwen3-VL-8B-Instruct",
  "gemma_31b": "google/gemma-4-31B-it",
  "gemma_26b": "google/gemma-4-26B-A4B-it",
  "qwen_27b": "Qwen/Qwen3.6-27B",
  "qwen_9b": "Qwen/Qwen3.5-9B",
  "gemma_e4b": "google/gemma-4-E4B-it",
}

# Mapping test names to functions
TEST_FUNCTIONS = {
  "mnist": test_mnist,
  "wb": test_wb,
  "chc": test_chc,
}


def main():
  parser = argparse.ArgumentParser(
    description="Run evaluation tests with a selected model."
  )

  parser.add_argument(
    "--model",
    type=str,
    default="qwen_vl",
    choices=AVAILABLE_MODELS.keys(),
    help=f"Model to use: {', '.join(AVAILABLE_MODELS.keys())}"
  )

  parser.add_argument(
    "--test",
    type=str,
    required=True,
    choices=TEST_FUNCTIONS.keys(),
    help="Which test to run (mnist, wb, chc)"
  )

  args = parser.parse_args()

  model_name = AVAILABLE_MODELS[args.model]

  print(f"Using model alias: {args.model}")
  print(f"Resolved model: {model_name}")
  print(f"Running test: {args.test}")

  test_fn = TEST_FUNCTIONS[args.test]
  test_fn(model_name=model_name)


if __name__ == "__main__":
  main()