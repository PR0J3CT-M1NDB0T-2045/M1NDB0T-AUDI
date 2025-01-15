import logging
import csv
import datetime
from pathlib import Path
from typing import Optional, List, Any

import torch
import torchaudio

from mmaudio.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate,
    load_video,
    make_video,
    setup_eval_logging
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

# ====================== Configuration Settings ====================== #

# Model variant to use. Options: 'small_16k', 'small_44k', 'medium_44k', 'large_44k', 'large_44k_v2'
MODEL_VARIANT = 'large_44k_v2'

# Path to the parent input directory containing series folders (e.g., 'MindBot-Series 1', 'MindBot-Series 2')
INPUT_DIR = Path('/path/to/input_dir')

# Authentication or API token
TOKEN = 'YOUR_TOKEN_HERE'

# Path to the parent output directory where processed outputs will be saved
OUTPUT_DIR = Path('/path/to/output_dir')

# Duration of the audio in seconds
AUDIO_DURATION = 8.0

# CFG strength (Control Guidance Factor)
CFG_STRENGTH = 4.5

# Number of inference steps
NUM_STEPS = 25

# Whether to mask away the clip frames
MASK_AWAY_CLIP = False

# Random seed for reproducibility
SEED = 42

# Whether to skip the video composite step
SKIP_VIDEO_COMPOSITE = False

# Whether to use full precision (float32). If False, uses bfloat16
FULL_PRECISION = False

# Path to the CSV file for tracking processing status
TRACKING_FILE = Path('processing_log.csv')

# Video file extensions to process
VIDEO_EXTENSIONS = ['*.mp4', '*.avi', '*.mov', '*.mkv']  # Add more if needed

# ==================================================================== #

# Enable TensorFloat-32 for faster computations on supported hardware
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def select_device() -> torch.device:
    """
    Selects the appropriate device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        log.warning('CUDA/MPS are not available, running on CPU')
    return device


def load_model(config: ModelConfig, device: torch.device, dtype: torch.dtype) -> MMAudio:
    """
    Loads the MMAudio model with the specified configuration.

    Args:
        config (ModelConfig): The model configuration.
        device (torch.device): The device to load the model on.
        dtype (torch.dtype): The data type for the model parameters.

    Returns:
        MMAudio: The loaded MMAudio model.
    """
    model = get_my_mmaudio(config.model_name).to(device, dtype).eval()
    try:
        weights = torch.load(config.model_path, map_location=device)
        model.load_weights(weights, weights_only=True)
        log.info(f'Loaded weights from {config.model_path}')
    except Exception as e:
        log.error(f'Failed to load weights from {config.model_path}: {e}')
        raise
    return model


def initialize_feature_utils(config: ModelConfig, device: torch.device, dtype: torch.dtype) -> FeaturesUtils:
    """
    Initializes the FeaturesUtils with the specified configuration.

    Args:
        config (ModelConfig): The model configuration.
        device (torch.device): The device to load the features utils on.
        dtype (torch.dtype): The data type for the features utils.

    Returns:
        FeaturesUtils: The initialized FeaturesUtils object.
    """
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=config.vae_path,
        synchformer_ckpt=config.synchformer_ckpt,
        enable_conditions=True,
        mode=config.mode,
        bigvgan_vocoder_ckpt=config.bigvgan_16k_path,
        need_vae_encoder=False
    ).to(device, dtype).eval()

    return feature_utils


def setup_rng(seed: int, device: torch.device) -> torch.Generator:
    """
    Sets up the random number generator with the specified seed.

    Args:
        seed (int): The seed for reproducibility.
        device (torch.device): The device for the generator.

    Returns:
        torch.Generator: The initialized random number generator.
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    return rng


def generate_audio(
    clip_frames: Optional[torch.Tensor],
    sync_frames: Optional[torch.Tensor],
    prompts: List[str],
    negative_prompts: List[str],
    feature_utils: FeaturesUtils,
    model: MMAudio,
    flow_matching: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float
) -> torch.Tensor:
    """
    Generates audio based on the provided inputs.

    Args:
        clip_frames (Optional[torch.Tensor]): Clip frames from the video.
        sync_frames (Optional[torch.Tensor]): Sync frames from the video.
        prompts (List[str]): List of input prompts.
        negative_prompts (List[str]): List of negative prompts.
        feature_utils (FeaturesUtils): Feature utilities.
        model (MMAudio): The MMAudio model.
        flow_matching (FlowMatching): The flow matching configuration.
        rng (torch.Generator): Random number generator.
        cfg_strength (float): CFG strength.

    Returns:
        torch.Tensor: The generated audio tensor.
    """
    audios = generate(
        clip_frames,
        sync_frames,
        prompts,
        negative_text=negative_prompts,
        feature_utils=feature_utils,
        net=model,
        fm=flow_matching,
        rng=rng,
        cfg_strength=cfg_strength
    )
    return audios.float().cpu()[0]


def save_audio(audio: torch.Tensor, save_path: Path, sampling_rate: int) -> None:
    """
    Saves the audio tensor to a file.

    Args:
        audio (torch.Tensor): The audio tensor.
        save_path (Path): The path to save the audio file.
        sampling_rate (int): The sampling rate of the audio.
    """
    try:
        torchaudio.save(save_path, audio, sampling_rate)
        log.info(f'Audio saved to {save_path}')
    except Exception as e:
        log.error(f'Failed to save audio to {save_path}: {e}')
        raise


def compose_video(video_info: Any, audio: torch.Tensor, save_path: Path, sampling_rate: int) -> None:
    """
    Composes the video with the generated audio.

    Args:
        video_info (Any): Information about the loaded video.
        audio (torch.Tensor): The generated audio tensor.
        save_path (Path): The path to save the composed video.
        sampling_rate (int): The sampling rate of the audio.
    """
    try:
        make_video(video_info, save_path, audio, sampling_rate=sampling_rate)
        log.info(f'Video saved to {save_path}')
    except Exception as e:
        log.error(f'Failed to compose video to {save_path}: {e}')
        raise


def initialize_tracking(tracking_file: Path) -> None:
    """
    Initializes the tracking CSV file with headers if it doesn't exist.

    Args:
        tracking_file (Path): The path to the tracking CSV file.
    """
    if not tracking_file.exists():
        with tracking_file.open('w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Series', 'Status', 'Timestamp', 'Token', 'Error'])
        log.info(f'Initialized tracking file at {tracking_file}')


def update_tracking(tracking_file: Path, series_name: str, status: str, token: str, error: Optional[str] = None) -> None:
    """
    Updates the tracking CSV file with the processing status of a series.

    Args:
        tracking_file (Path): The path to the tracking CSV file.
        series_name (str): The name of the series.
        status (str): The processing status ('Success', 'Failed', or 'No Videos Found').
        token (str): The token used for processing.
        error (Optional[str]): Error message if any.
    """
    timestamp = datetime.datetime.now().isoformat()
    with tracking_file.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([series_name, status, timestamp, token, error or ''])
    log.info(f'Updated tracking for series "{series_name}" with status "{status}"')


def process_series(
    series_path: Path,
    output_dir: Path,
    token: str,
    model: MMAudio,
    feature_utils: FeaturesUtils,
    flow_matching: FlowMatching,
    rng: torch.Generator,
    seq_cfg: Any,
    tracking_file: Path
) -> None:
    """
    Processes a single series folder.

    Args:
        series_path (Path): Path to the series folder.
        output_dir (Path): Path to the output series folder.
        token (str): Authentication or API token.
        model (MMAudio): The MMAudio model.
        feature_utils (FeaturesUtils): Feature utilities.
        flow_matching (FlowMatching): The flow matching configuration.
        rng (torch.Generator): Random number generator.
        seq_cfg (Any): Sequence configuration.
        tracking_file (Path): Path to the tracking CSV file.
    """
    series_name = series_path.name
    log.info(f'Starting processing for series: {series_name}')
    try:
        # Gather all video files with the specified extensions
        video_files = []
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(series_path.glob(ext))

        if not video_files:
            log.warning(f'No video files found in series "{series_name}". Skipping.')
            update_tracking(tracking_file, series_name, 'No Videos Found', token)
            return

        for video_file in video_files:
            log.info(f'Processing video file: {video_file}')
            try:
                video_info = load_video(video_file, AUDIO_DURATION)
                clip_frames = None if MASK_AWAY_CLIP else video_info.clip_frames.unsqueeze(0)
                sync_frames = video_info.sync_frames.unsqueeze(0)
                duration = video_info.duration_sec

                # Update sequence lengths based on the configuration
                seq_cfg.duration = duration
                model.update_seq_lengths(
                    seq_cfg.latent_seq_len,
                    seq_cfg.clip_seq_len,
                    seq_cfg.sync_seq_len
                )

                # Log prompts
                log.info(f'Prompt: "{PROMPT}"')
                log.info(f'Negative Prompt: "{NEGATIVE_PROMPT}"')

                # Generate audio
                audio = generate_audio(
                    clip_frames,
                    sync_frames,
                    [PROMPT],
                    [NEGATIVE_PROMPT],
                    feature_utils,
                    model,
                    flow_matching,
                    rng,
                    CFG_STRENGTH
                )

                # Determine save paths
                audio_save_path = output_dir / f'{video_file.stem}.flac'
                save_audio(audio, audio_save_path, seq_cfg.sampling_rate)

                if not SKIP_VIDEO_COMPOSITE:
                    video_save_path = output_dir / f'{video_file.stem}_composed.mp4'
                    compose_video(video_info, audio, video_save_path, seq_cfg.sampling_rate)

            except Exception as video_e:
                log.error(f'Failed to process video "{video_file}": {video_e}')
                update_tracking(tracking_file, series_name, 'Failed',
                                token, f'Video {video_file.name}: {video_e}')
                continue  # Continue with next video

        # Update tracking as success
        update_tracking(tracking_file, series_name, 'Success', token)
    except Exception as e:
        # Update tracking as failed with error message
        update_tracking(tracking_file, series_name, 'Failed', token, str(e))
        log.error(f'Failed to process series "{series_name}": {e}')


# ============================ MAIN FUNCTION ============================ #

def main() -> None:
    """
    The main function to execute the MMAudio evaluation for multiple series.
    """
    setup_eval_logging()

    # Validate model variant
    if MODEL_VARIANT not in all_model_cfg:
        log.error(f'Unknown model variant: {MODEL_VARIANT}')
        sys.exit(1)

    model_config: ModelConfig = all_model_cfg[MODEL_VARIANT]
    model_config.download_if_needed()
    seq_cfg = model_config.seq_cfg

    # Prepare output parent directory
    output_parent_dir: Path = OUTPUT_DIR.expanduser()
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tracking
    initialize_tracking(TRACKING_FILE)

    # Select device and dtype
    device = select_device()
    dtype = torch.float32 if FULL_PRECISION else torch.bfloat16

    # Load the model
    model = load_model(model_config, device, dtype)

    # Initialize feature utilities
    feature_utils = initialize_feature_utils(model_config, device, dtype)

    # Set up random number generator
    rng = setup_rng(SEED, device)

    # Initialize flow matching
    flow_matching = FlowMatching(
        min_sigma=0,
        inference_mode='euler',
        num_steps=NUM_STEPS
    )

    # Iterate through each series folder in the input directory
    input_dir: Path = INPUT_DIR.expanduser()
    if not input_dir.exists() or not input_dir.is_dir():
        log.error(f'Input directory "{input_dir}" does not exist or is not a directory.')
        sys.exit(1)

    for series_path in input_dir.iterdir():
        if series_path.is_dir():
            # Define corresponding output directory for the series
            output_dir = output_parent_dir / series_path.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Process the series
            process_series(
                series_path=series_path,
                output_dir=output_dir,
                token=TOKEN,
                model=model,
                feature_utils=feature_utils,
                flow_matching=flow_matching,
                rng=rng,
                seq_cfg=seq_cfg,
                tracking_file=TRACKING_FILE
            )

    # Log memory usage if CUDA is used
    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated(device) / (2**30)
        log.info(f'Maximum CUDA memory allocated: {max_memory:.2f} GB')
    else:
        log.info('Memory usage logging is only available for CUDA devices.')


# ============================= PROMPTS =============================== #

# Define your prompts here
PROMPT = 'Your positive prompt here'
NEGATIVE_PROMPT = 'Your negative prompt here'

# ==================================================================== #


if __name__ == '__main__':
    main()
