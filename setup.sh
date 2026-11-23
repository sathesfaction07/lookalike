#!/usr/bin/env bash
set -e

VENV_DIR=".venv"

detect_os() { case "$(uname -s)" in Linux*) OS="linux";; Darwin*) OS="macos";; MINGW*|MSYS*|CYGWIN*) OS="windows";; *) OS="unknown";; esac; echo "Detected OS: $OS"; }
find_python() { for cmd in python3.12 python3.11 python3 python; do if command -v "$cmd" &>/dev/null; then PYTHON="$cmd"; echo "Using: $PYTHON"; return 0; fi; done; echo "ERROR: Python 3.11+ not found."; exit 1; }

setup_venv() {
    [ -d "$VENV_DIR" ] && echo "Venv exists — reusing." || $PYTHON -m venv "$VENV_DIR"
    if [ "$OS" = "windows" ]; then source "$VENV_DIR/Scripts/activate"; else source "$VENV_DIR/bin/activate"; fi
    pip install --upgrade pip --quiet && pip install -e ".[dev]" --quiet
    echo "Dependencies installed."
}

select_backend() {
    echo ""
    echo "  Lookalike supports multiple embedding backends:"
    echo "  1) Local only (EfficientNet-B0 — no cloud needed)"
    echo "  2) AWS Titan Multimodal Embeddings (better for product images)"
    echo "  3) Google Vertex AI Multimodal Embeddings"
    read -p "Selection [1/2/3]: " CHOICE
    case "$CHOICE" in
        2) setup_aws;;
        3) setup_gcp;;
        *) echo "Using local backend. No cloud setup needed.";;
    esac
}

setup_aws() {
    read -p "AWS Account ID: " ACCT
    read -p "Region [us-east-1]: " REGION; REGION=${REGION:-us-east-1}
    ROLE="Lookalike-AppRole"
    POLICY="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"bedrock:InvokeModel\"],\"Resource\":\"arn:aws:bedrock:${REGION}::foundation-model/amazon.titan-embed-image-v1\"}]}"
    TRUST="{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"arn:aws:iam::${ACCT}:root\"},\"Action\":\"sts:AssumeRole\"}]}"

    aws iam get-role --role-name "$ROLE" &>/dev/null 2>&1 || aws iam create-role --role-name "$ROLE" --assume-role-policy-document "$TRUST" --output text --query 'Role.Arn'
    aws iam put-role-policy --role-name "$ROLE" --policy-name "Lookalike-Bedrock" --policy-document "$POLICY" 2>/dev/null
    pip install "lookalike[titan]" --quiet
    echo "AWS_DEFAULT_REGION=${REGION}" > .env
    echo "AWS Titan backend configured."
}

setup_gcp() {
    read -p "GCP Project ID: " PROJECT
    gcloud config set project "$PROJECT" --quiet
    gcloud services enable aiplatform.googleapis.com --quiet 2>/dev/null
    pip install "lookalike[vertex]" --quiet
    echo "GCP_PROJECT=${PROJECT}" > .env
    echo "Vertex AI backend configured."
}

main() {
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║            Lookalike — Setup                              ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    detect_os; find_python; setup_venv; select_backend
    echo ""
    echo "Ready."
    echo "  CLI:  lookalike --input data/sample_input.csv --output results/"
    echo "  Web:  uvicorn image_cluster.api.app:app --port 8000"
}

main "$@"
