import argparse
import os
from dotenv import load_dotenv
from judge_wrapper import *
from run_algorithmic import run_algorithmic_judges

# Load environment variables from .env file
load_dotenv()

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        dest="input",
        help="Path containing file to run",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        dest="output",
        help="Path to save output",
    )
    parser.add_argument(
        "-e",
        "--algorithmic_evaluators",
        type=str,
        dest="evaluators",
        help="Algorithmic Evaluators configuration file",
        default="scripts/evaluation/config.yaml",
    )
    parser.add_argument(
        "--provider", 
        type=str,
        required=True,
        dest="provider", 
        choices=["openai", "azure", "hf"],
        help="Provider to use for LLM judges",
    )
    parser.add_argument(
        "--judge_model", 
        type=str, 
        dest="judge_model",
        help="Hugging Face model name (required if provider=hf)"
    )
    parser.add_argument(
        "--openai_key", 
        type=str, 
        help="OpenAI/Azure Key (required if provider=openai/azure)"
    )
    parser.add_argument(
        "--azure_host", 
        type=str, 
        help="Azure OpenAI endpoint (required if provider=azure)"
    )
    return parser

if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()
    
    run_algorithmic_judges(args.evaluators, args.input, args.output)
    
    if args.provider == "openai":
        if not args.openai_key and not os.environ.get("OPENAI_API_KEY"):
            parser.error("--provider openai requires --openai_key or OPENAI_API_KEY environment variable")
        if args.openai_key:
            os.environ["OPENAI_API_KEY"] = args.openai_key
        judge_model = args.judge_model or "gpt-4o-mini"

    elif args.provider == "azure":
        if not args.openai_key or not args.azure_host:
            parser.error("--provider azure requires --openai_key and --azure_host")
        os.environ["AZURE_OPENAI_API_KEY"] = args.openai_key
        os.environ["OPENAI_AZURE_HOST"] = args.azure_host
        judge_model = args.judge_model or "gpt-4o-mini-2024-07-18"

    else:  # hf
        if not args.judge_model:
            parser.error(f"--provider {args.provider} requires --judge_model")
        judge_model = args.judge_model
    
    
    if args.provider in ["openai", "azure"]:
        run_idk_judge(args.provider, args.output, args.output)
        run_ragas_judges_openai(args.output, args.output, args.openai_key, args.azure_host, provider=args.provider)
        run_radbench_judge(args.provider, args.output, args.output)
        
        get_idk_conditioned_metrics(args.output, args.output)
    else:  # hf
        run_idk_judge(judge_model, args.output, args.output)
        run_ragas_judges_local(judge_model, args.output, args.output)
        run_radbench_judge(judge_model, args.output, args.output)
        
        get_idk_conditioned_metrics(args.output, args.output)