import click
import json
import logging
import pandas as pd
import sys
from masktab.inference.engine import predict
from masktab.data.loader import load_csv

logger = logging.getLogger(__name__)

@click.command()
@click.option("--input-csv", required=True, type=click.Path(exists=True), help="Path to input CSV for inference.")
@click.option("--task-name", required=True, type=str, help="Name of the task to run inference for.")
@click.option("--db", required=True, type=click.Path(), help="Path to SQLite database.")
@click.option("--confidence-threshold", type=float, default=0.60, help="Threshold below which classification predictions are marked UNCERTAIN.")
@click.option("--disable-drift-check", is_flag=True, help="Disable PSI drift checking.")
@click.option("--output-json", type=click.Path(), default="inference_output.json", help="Path to write the detailed JSON output.")
def infer(input_csv, task_name, db, confidence_threshold, disable_drift_check, output_json):
    """Run inference for a specific task on a CSV dataset."""
    try:
        # Load CSV using Phase 1 loader to ensure consistent parsing, but predict takes DataFrame
        df, _ = load_csv(input_csv)
        
        results = predict(
            input_data=df,
            task_name=task_name,
            db_path=db,
            cache=None, # CLI doesn't persist cache across runs usually, or we could pass one but it's single batch anyway
            confidence_threshold=confidence_threshold,
            enable_drift_check=not disable_drift_check
        )
        
        # Write to JSON
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
            
        # Compute summary
        row_count = len(results)
        uncertain_count = sum(1 for r in results if r.get("predicted_label_status") == "UNCERTAIN")
        
        # Drift status
        drift_detected = False
        drift_skipped = False
        for r in results:
            if r.get("drift_detected") is True:
                drift_detected = True
                break
            elif r.get("drift_detected") is None and any("skipped" in note for note in r.get("notes", [])):
                drift_skipped = True
                
        if drift_detected:
            drift_status = "DETECTED"
        elif drift_skipped:
            drift_status = "SKIPPED"
        elif disable_drift_check:
            drift_status = "DISABLED"
        else:
            drift_status = "NOT DETECTED"

        click.echo("--- Inference Summary ---")
        click.echo(f"Processed Rows: {row_count}")
        click.echo(f"Uncertain Predictions: {uncertain_count}")
        click.echo(f"Drift Status: {drift_status}")
        click.echo(f"Full output written to: {output_json}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    infer()
