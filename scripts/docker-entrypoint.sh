#!/bin/bash
set -e

# Print initial environment values (before loading .env)
echo "Starting with these environment variables:"
echo "APP_ENV: ${APP_ENV:-development}"
echo "Initial Database Host: $( [[ -n ${POSTGRES_HOST:-${DB_HOST:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Initial Database Port: $( [[ -n ${POSTGRES_PORT:-${DB_PORT:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Initial Database Name: $( [[ -n ${POSTGRES_DB:-${DB_NAME:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Initial Database User: $( [[ -n ${POSTGRES_USER:-${DB_USER:-}} ]] && echo 'set' || echo 'Not set' )"

# Load environment variables from the appropriate .env file
if [ -f ".env.${APP_ENV}" ]; then
    echo "Loading environment from .env.${APP_ENV}"
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue

        # Extract the key
        key=$(echo "$line" | cut -d '=' -f 1)

        # Only set if not already set in environment
        if [[ -z "${!key}" ]]; then
            export "$line"
        else
            echo "Keeping existing value for $key"
        fi
    done <".env.${APP_ENV}"
elif [ -f ".env" ]; then
    echo "Loading environment from .env"
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue

        # Extract the key
        key=$(echo "$line" | cut -d '=' -f 1)

        # Only set if not already set in environment
        if [[ -z "${!key}" ]]; then
            export "$line"
        else
            echo "Keeping existing value for $key"
        fi
    done <".env"
else
    echo "Warning: No .env file found. Using system environment variables."
fi

# Check required sensitive environment variables
# Changed default provider to OpenAI (Ollama commented out)
LLM_PROVIDER=${LLM_PROVIDER:-openai}
LLM_PROVIDER=$(echo "$LLM_PROVIDER" | tr '[:upper:]' '[:lower:]')

required_vars=("JWT_SECRET_KEY")
missing_vars=()

# Check LLM provider-specific requirements
if [[ "$LLM_PROVIDER" == "openai" ]]; then
    required_vars+=("OPENAI_API_KEY")
    echo "LLM Provider: OpenAI - Checking for OPENAI_API_KEY"
# Ollama provider commented out - using OpenAI instead
# elif [[ "$LLM_PROVIDER" == "ollama" ]]; then
#     echo "LLM Provider: Ollama - Checking for OLLAMA_BASE_URL"
#     if [[ -z "${OLLAMA_BASE_URL}" ]]; then
#         echo "Warning: OLLAMA_BASE_URL not set, using default: http://localhost:11434"
#         export OLLAMA_BASE_URL="http://localhost:11434"
#     fi
#     if [[ -z "${DEFAULT_LLM_MODEL}" ]]; then
#         echo "Warning: DEFAULT_LLM_MODEL not set, using default: qwen2.5:7b-instruct-q4_K_M"
#         export DEFAULT_LLM_MODEL="qwen2.5:7b-instruct-q4_K_M"
#     fi
else
    echo "Warning: Unknown LLM_PROVIDER: $LLM_PROVIDER. Defaulting to OpenAI."
    export LLM_PROVIDER="openai"
fi

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "ERROR: The following required environment variables are missing:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo "Please provide these variables through environment or .env files."
    exit 1
fi

# Print final environment info
echo -e "\nFinal environment configuration:"
echo "Environment: ${APP_ENV:-development}"

echo "Database Host: $( [[ -n ${POSTGRES_HOST:-${DB_HOST:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Database Port: $( [[ -n ${POSTGRES_PORT:-${DB_PORT:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Database Name: $( [[ -n ${POSTGRES_DB:-${DB_NAME:-}} ]] && echo 'set' || echo 'Not set' )"
echo "Database User: $( [[ -n ${POSTGRES_USER:-${DB_USER:-}} ]] && echo 'set' || echo 'Not set' )"

echo "LLM Provider: ${LLM_PROVIDER:-openai}"
echo "LLM Model: ${DEFAULT_LLM_MODEL:-Not set}"
# Ollama check commented out - using OpenAI instead
# if [[ "$LLM_PROVIDER" == "ollama" ]]; then
#     echo "Ollama Base URL: ${OLLAMA_BASE_URL:-Not set}"
# fi
echo "Debug Mode: ${DEBUG:-false}"

# Run database migrations if necessary
# e.g., alembic upgrade head

# Execute the CMD
exec "$@"
