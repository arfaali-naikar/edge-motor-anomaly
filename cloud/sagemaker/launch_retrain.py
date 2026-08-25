"""
Kick off a SageMaker training job against the spooled telemetry.

    python cloud/sagemaker/launch_retrain.py --dry-run
    python cloud/sagemaker/launch_retrain.py --commit

Costs money when you --commit. ml.m5.large is about $0.13/hr in eu-west-2 and
this job takes a couple of minutes, so a run is pennies -- but it is not zero,
and a forgotten job is not pennies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from motor_anomaly.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--commit", action="store_true")
    ap.add_argument("--instance-type", default="ml.m5.large")
    ap.add_argument("--feature-version", type=int, default=3)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bucket, prefix, region = cfg["cloud"]["bucket"], cfg["cloud"]["prefix"], cfg["cloud"]["region"]
    role = cfg["cloud"]["role_arn"]
    s3_input = f"s3://{bucket}/{prefix}/feature_version={args.feature_version}/"

    print(f"input   {s3_input}")
    print(f"output  s3://{bucket}/models/")
    print(f"role    {role or '(not set)'}")
    print(f"region  {region}   instance {args.instance_type}")

    if not args.commit:
        print("\ndry run -- pass --commit to actually launch")
        return 0
    if not role:
        print("\ncloud.role_arn is empty; run `terraform output -raw sagemaker_role_arn` first")
        return 1
    if bucket.startswith("CHANGEME"):
        print("\nset a real bucket name in config/default.yaml first")
        return 1

    from sagemaker.inputs import TrainingInput
    from sagemaker.tensorflow import TensorFlow

    est = TensorFlow(
        entry_point="entry_point.py",
        source_dir=str(Path(__file__).parent),
        # Ship src/ so the container can `import motor_anomaly` -- this is what
        # keeps the cloud model identical to the edge one.
        dependencies=[str(Path(__file__).resolve().parents[2] / "src" / "motor_anomaly")],
        role=role,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version="2.16",
        py_version="py310",
        output_path=f"s3://{bucket}/models/",
        base_job_name="motor-anomaly-retrain",
        hyperparameters={"epochs": 150, "feature-version": args.feature_version},
    )
    est.fit({"training": TrainingInput(s3_input, distribution="FullyReplicated")})
    print(f"\nmodel artifact: {est.model_data}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
