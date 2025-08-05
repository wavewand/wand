#!/usr/bin/env python3
"""
Script to generate Python gRPC code from Protocol Buffer definitions.
"""

import subprocess
import sys
from pathlib import Path


def generate_grpc_code():
    """Generate Python gRPC code from .proto files."""

    # Get project root
    project_root = Path(__file__).parent.parent
    proto_dir = project_root / "protos"
    output_dir = project_root / "generated"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Create __init__.py files
    (output_dir / "__init__.py").touch()

    # Find all .proto files
    proto_files = list(proto_dir.glob("*.proto"))

    if not proto_files:
        print("No .proto files found in protos/ directory")
        return False

    print(f"Found {len(proto_files)} proto files: {[f.name for f in proto_files]}")

    # Generate Python code for each proto file
    for proto_file in proto_files:
        print(f"Generating code for {proto_file.name}...")

        # Run protoc command
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"--proto_path={proto_dir}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            str(proto_file),
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Generated code for {proto_file.name}")

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error generating code for {proto_file.name}:")
            print(f"    {e.stderr}")
            return False

    # Fix import paths in generated files
    _fix_imports(output_dir)

    print("✓ gRPC code generation completed successfully")
    return True


def _fix_imports(output_dir: Path):
    """Fix relative imports in generated gRPC files."""

    # Find all generated _pb2_grpc.py files
    grpc_files = list(output_dir.glob("*_pb2_grpc.py"))

    for grpc_file in grpc_files:
        content = grpc_file.read_text()

        # Fix imports - change relative to absolute
        pb2_name = grpc_file.name.replace("_grpc.py", ".py")
        old_import = f"import {pb2_name.replace('.py', '')}"
        new_import = f"from . import {pb2_name.replace('.py', '')}"

        if old_import in content and new_import not in content:
            content = content.replace(old_import, new_import)
            grpc_file.write_text(content)
            print(f"  ✓ Fixed imports in {grpc_file.name}")


if __name__ == "__main__":
    success = generate_grpc_code()
    sys.exit(0 if success else 1)
