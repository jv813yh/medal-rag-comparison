# Error Recovery Workflow

1.  **Detection**: Orchestrator detects tool failure or exception.
2.  **Diagnosis**: Repair Agent inspects logs and tool output.
3.  **Correction**:
    *   If dependency issue: Update `requirements.txt`.
    *   If logic issue: Modify tool code.
    *   If configuration issue: Check `.env` or `config.py`.
4.  **Verification**: Re-run the failing step.
5.  **Documentation**: Update workflows if the approach needs to change permanently.
