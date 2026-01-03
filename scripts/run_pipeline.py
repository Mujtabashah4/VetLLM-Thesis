#!/usr/bin/env python3
"""
VetLLM Complete Pipeline Script
Orchestrates data validation, training, and inference for VetLLM
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_vetllm import VetLLMTrainer, VetLLMConfig
from scripts.inference import VetLLMInference
from scripts.validate_data import DataValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VetLLMPipeline:
    """Complete VetLLM Pipeline"""
    
    def __init__(self, config_path: str = None):
        """Initialize pipeline"""
        self.config_path = config_path
        self.results = {
            "pipeline_start": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": []
        }
    
    def validate_data(self, data_path: str) -> bool:
        """
        Validate training data.
        
        Args:
            data_path: Path to data file
        
        Returns:
            True if validation passed, False otherwise
        """
        logger.info("="*70)
        logger.info("STEP 1: DATA VALIDATION")
        logger.info("="*70)
        
        try:
            validator = DataValidator(data_path)
            is_valid, stats = validator.validate()
            
            self.results["data_validation"] = {
                "file": data_path,
                "valid": is_valid,
                "stats": stats
            }
            self.results["steps_completed"].append("data_validation")
            
            if is_valid:
                logger.info(" Data validation passed!")
                return True
            else:
                logger.error(" Data validation failed!")
                self.results["errors"].append("Data validation failed")
                return False
                
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            self.results["errors"].append(f"Data validation error: {str(e)}")
            return False
    
    def train(
        self,
        model_name: str,
        data_path: str,
        val_data_path: str = None,
        output_dir: str = "models/vetllm-finetuned",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        use_8bit: bool = True,
        use_wandb: bool = False,
    ):
        """
        Train the model.
        
        Args:
            model_name: Base model name
            data_path: Training data path
            val_data_path: Validation data path (optional)
            output_dir: Output directory for trained model
            epochs: Number of training epochs
            batch_size: Per-device batch size
            learning_rate: Learning rate
            use_8bit: Use 8-bit quantization
            use_wandb: Use Weights & Biases logging
        """
        logger.info("="*70)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("="*70)
        
        try:
            # Create config
            config = VetLLMConfig(
                model_name=model_name,
                data_path=data_path,
                val_data_path=val_data_path or "",
                output_dir=output_dir,
                num_epochs=epochs,
                per_device_batch_size=batch_size,
                learning_rate=learning_rate,
                use_8bit=use_8bit,
                use_wandb=use_wandb,
            )
            
            # Create trainer
            trainer = VetLLMTrainer(config)
            
            # Prepare datasets
            logger.info("Preparing datasets...")
            train_dataset = trainer.data_processor.prepare_dataset(data_path)
            
            eval_dataset = None
            if val_data_path and os.path.exists(val_data_path):
                eval_dataset = trainer.data_processor.prepare_dataset(val_data_path)
            
            # Train
            train_result = trainer.train(train_dataset, eval_dataset)
            
            self.results["training"] = {
                "model_name": model_name,
                "output_dir": output_dir,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "training_loss": train_result.training_loss,
                "training_steps": train_result.global_step,
            }
            self.results["steps_completed"].append("training")
            
            logger.info(" Training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            self.results["errors"].append(f"Training error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def inference(
        self,
        model_path: str,
        base_model_name: str = None,
        clinical_note: str = None,
        input_file: str = None,
        output_file: str = None,
        extract_codes: bool = True,
    ):
        """
        Run inference on clinical notes.
        
        Args:
            model_path: Path to trained model
            base_model_name: Base model name if using LoRA
            clinical_note: Single clinical note (or use input_file)
            input_file: JSON file with clinical notes
            output_file: Output file for predictions
            extract_codes: Extract SNOMED codes from predictions
        """
        logger.info("="*70)
        logger.info("STEP 3: INFERENCE")
        logger.info("="*70)
        
        try:
            # Initialize inference engine
            inference = VetLLMInference(
                model_path=model_path,
                base_model_name=base_model_name,
            )
            
            # Prepare input
            if clinical_note:
                notes = [{"note": clinical_note}]
            elif input_file:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], dict):
                        notes = data
                    else:
                        notes = [{"note": note} for note in data]
                else:
                    raise ValueError("Input file must contain a list")
            else:
                raise ValueError("Either clinical_note or input_file must be provided")
            
            # Run predictions
            results = []
            for item in notes:
                note = item.get("note", item) if isinstance(item, dict) else item
                prediction = inference.predict(note)
                
                result = {
                    "note": note,
                    "prediction": prediction
                }
                
                if extract_codes:
                    codes = inference.extract_snomed_codes(prediction)
                    result["snomed_codes"] = codes
                
                results.append(result)
            
            # Save results
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                logger.info(f" Results saved to {output_file}")
            
            self.results["inference"] = {
                "model_path": model_path,
                "num_notes": len(notes),
                "results": results
            }
            self.results["steps_completed"].append("inference")
            
            logger.info(" Inference completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            self.results["errors"].append(f"Inference error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_report(self, output_file: str = "pipeline_report.json"):
        """Save pipeline execution report"""
        self.results["pipeline_end"] = datetime.now().isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f" Pipeline report saved to {output_file}")

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(
        description="VetLLM Complete Pipeline - Data Validation, Training, and Inference"
    )
    
    # Pipeline steps
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data, don't train"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train model, skip validation and inference"
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Only run inference, skip validation and training"
    )
    
    # Data paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="processed_data/all_processed_data.json",
        help="Training data path"
    )
    parser.add_argument(
        "--val-data-path",
        type=str,
        default=None,
        help="Validation data path (optional)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="wxjiao/alpaca-7b",
        help="Base model name"
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
        default=None,
        help="Base model name for LoRA inference (defaults to model-name)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/vetllm-finetuned",
        help="Output directory for trained model"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--no-8bit",
        action="store_true",
        help="Disable 8-bit quantization"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    # Inference configuration
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model for inference"
    )
    parser.add_argument(
        "--clinical-note",
        type=str,
        default=None,
        help="Clinical note for inference"
    )
    parser.add_argument(
        "--inference-input-file",
        type=str,
        default=None,
        help="Input file with clinical notes for inference"
    )
    parser.add_argument(
        "--inference-output-file",
        type=str,
        default=None,
        help="Output file for inference results"
    )
    
    # Report
    parser.add_argument(
        "--report-file",
        type=str,
        default="pipeline_report.json",
        help="Pipeline report output file"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VetLLMPipeline()
    
    success = True
    
    # Step 1: Validate data (unless inference-only)
    if not args.inference_only:
        if not pipeline.validate_data(args.data_path):
            logger.error("Data validation failed. Exiting.")
            sys.exit(1)
    
    # Step 2: Train model (unless validate-only or inference-only)
    if not args.validate_only and not args.inference_only:
        if not pipeline.train(
            model_name=args.model_name,
            data_path=args.data_path,
            val_data_path=args.val_data_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_8bit=not args.no_8bit,
            use_wandb=args.wandb,
        ):
            logger.error("Training failed. Exiting.")
            success = False
    
    # Step 3: Run inference (unless validate-only or train-only)
    if not args.validate_only and not args.train_only:
        model_path = args.model_path or args.output_dir
        base_model_name = args.base_model_name or args.model_name
        
        if args.clinical_note or args.inference_input_file:
            if not pipeline.inference(
                model_path=model_path,
                base_model_name=base_model_name,
                clinical_note=args.clinical_note,
                input_file=args.inference_input_file,
                output_file=args.inference_output_file,
            ):
                logger.error("Inference failed.")
                success = False
        else:
            logger.info("Skipping inference (no clinical note or input file provided)")
    
    # Save report
    pipeline.save_report(args.report_file)
    
    if success:
        logger.info("\n" + "="*70)
        logger.info(" PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        sys.exit(0)
    else:
        logger.error("\n" + "="*70)
        logger.error(" PIPELINE COMPLETED WITH ERRORS")
        logger.error("="*70)
        sys.exit(1)

if __name__ == "__main__":
    main()

