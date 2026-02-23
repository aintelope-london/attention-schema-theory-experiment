import os
import glob


def select_checkpoint(cfg, agent_id, role):
    checkpoints = []

    # Find old model if user defined one
    old_checkpoint = cfg.dl_params.start_with_model
    if len(checkpoints) == 0 and old_checkpoint:
        checkpoints = glob.glob(old_checkpoint)

    # Overwrite possible old checkpoint if we are already into training
    # Check for previous model in checkpoints
    if cfg.dl_params.use_previous_model:
        checkpoint_dir = os.path.normpath(cfg.addresses.pipeline_dir + "checkpoints/")
        checkpoint_filename = f"*{agent_id}_{role}*"
        checkpoint = os.path.join(checkpoint_dir, checkpoint_filename)
        checkpoints.extend(glob.glob(checkpoint))

    # Return the latest model we have
    if len(checkpoints) > 0:
        return max(checkpoints, key=os.path.getctime)
    else:
        return False
