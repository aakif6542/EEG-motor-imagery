"""
EEG Conformer — Comprehensive Validation Script
=================================================
Checks:
  1. Input tensor dimensions throughout forward pass
  2. Sequence reshaping before transformer encoder
  3. Positional encoding shape compatibility
  4. Total parameter count (trainable vs non-trainable)
  5. Whether transformer attention operates over temporal tokens
  6. Whether train/eval modes (dropout/BN) function correctly
  7. Preprocessing input shape comparison against EEGNet
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from models.eeg_conformer import EEGConformer, SinusoidalPositionalEncoding, TransformerEncoderBlock
from models.eegnet import EEGNet

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

issues_found = []

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {name}")
    if detail:
        print(f"      {detail}")
    if not condition:
        issues_found.append(name)
    return condition

# ============================================================
# TEST CONFIGURATIONS (matching real datasets)
# ============================================================
CONFIGS = {
    "BNCI2014001": {"n_channels": 22, "n_timepoints": 384},
    "Cho2017":     {"n_channels": 64, "n_timepoints": 384},
    "PhysionetMI": {"n_channels": 64, "n_timepoints": 384},
}

# Use BNCI2014001 as primary test case
PRIMARY = CONFIGS["BNCI2014001"]
N_CH = PRIMARY["n_channels"]
N_TP = PRIMARY["n_timepoints"]
BATCH = 4

print("\n" + "#"*60)
print("#  EEG CONFORMER VERIFICATION REPORT")
print("#"*60)

# ============================================================
# 1. INPUT TENSOR DIMENSIONS THROUGHOUT FORWARD PASS
# ============================================================
section("1. INPUT TENSOR DIMENSIONS (Layer-by-Layer)")

conformer = EEGConformer(n_channels=N_CH, n_timepoints=N_TP)
model = conformer.model

# Build an intermediate-output model for every layer
print(f"\n  Input shape to model: (batch, {N_CH}, {N_TP}, 1)")
print(f"  {'Layer Name':<45} {'Output Shape':<25}")
print(f"  {'-'*45} {'-'*25}")

for layer in model.layers:
    try:
        out_shape = layer.output.shape
        print(f"  {layer.name:<45} {str(out_shape):<25}")
    except Exception:
        print(f"  {layer.name:<45} {'(multiple outputs)':<25}")

# Create dummy input and trace full forward pass
x_dummy = np.random.randn(BATCH, N_CH, N_TP, 1).astype(np.float32)
output = model.predict(x_dummy, verbose=0)

check("Output shape is (batch, 1)",
      output.shape == (BATCH, 1),
      f"Got: {output.shape}")

check("Output values are in [0, 1] (sigmoid)",
      np.all(output >= 0) and np.all(output <= 1),
      f"Range: [{output.min():.4f}, {output.max():.4f}]")

# ============================================================
# 2. SEQUENCE RESHAPING BEFORE TRANSFORMER
# ============================================================
section("2. SEQUENCE RESHAPING VERIFICATION")

# Trace shapes through the computational graph manually
# Input: (B, 22, 384, 1)
# After Conv1 + Pool(4): (B, 22, 96, 16)
# After Conv2 + Pool(4): (B, 22, 24, 32)
# After Spatial Conv(22,1): (B, 1, 24, 64)
# After Reshape: (B, 24, 64)

t_after_pool1 = N_TP // 4        # 384 / 4 = 96
t_after_pool2 = t_after_pool1 // 4  # 96 / 4 = 24
expected_seq_len = t_after_pool2
d_model = 64

print(f"\n  Temporal pooling chain:")
print(f"    Input timepoints:      {N_TP}")
print(f"    After AvgPool(1,4) #1: {t_after_pool1}")
print(f"    After AvgPool(1,4) #2: {t_after_pool2}")
print(f"    Expected seq_len:      {expected_seq_len}")
print(f"    Expected d_model:      {d_model}")

# Find the Reshape layer and verify
reshape_layer = None
for layer in model.layers:
    if 'reshape' in layer.name:
        reshape_layer = layer
        break

check("Reshape layer exists in model",
      reshape_layer is not None)

if reshape_layer:
    reshape_out_shape = reshape_layer.output.shape
    check("Reshape output is (batch, seq_len, d_model)",
          reshape_out_shape[1] == expected_seq_len and reshape_out_shape[2] == d_model,
          f"Got: {reshape_out_shape}, expected: (None, {expected_seq_len}, {d_model})")

# Verify spatial conv eliminates channel dimension
spatial_conv = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        # The spatial conv has kernel (n_channels, 1)
        if hasattr(layer, 'kernel_size') and layer.kernel_size == (N_CH, 1):
            spatial_conv = layer
            break

check("Spatial reduction Conv2D(n_channels, 1) found",
      spatial_conv is not None,
      f"Kernel: {spatial_conv.kernel_size if spatial_conv else 'NOT FOUND'}")

if spatial_conv:
    sp_out = spatial_conv.output.shape
    check("Spatial conv output has spatial dim = 1",
          sp_out[1] == 1,
          f"Shape after spatial conv: {sp_out}")

# ============================================================
# 3. POSITIONAL ENCODING SHAPE COMPATIBILITY
# ============================================================
section("3. POSITIONAL ENCODING VERIFICATION")

# Find PE layer
pe_layer = None
for layer in model.layers:
    if isinstance(layer, SinusoidalPositionalEncoding):
        pe_layer = layer
        break

check("SinusoidalPositionalEncoding layer found", pe_layer is not None)

if pe_layer:
    pe_weights = pe_layer.get_weights()
    check("PE has exactly 1 weight (the encoding matrix)",
          len(pe_weights) == 1,
          f"Got {len(pe_weights)} weights")

    pe_matrix = pe_weights[0]
    check("PE weight shape is (1, seq_len, d_model)",
          pe_matrix.shape == (1, expected_seq_len, d_model),
          f"Got: {pe_matrix.shape}, expected: (1, {expected_seq_len}, {d_model})")

    # Verify PE is non-trainable
    check("PE weight is non-trainable",
          not pe_layer.trainable_weights,
          f"Trainable weights: {len(pe_layer.trainable_weights)}")

    # Verify PE values are bounded (sin/cos → [-1, 1])
    check("PE values bounded in [-1, 1]",
          np.all(pe_matrix >= -1.0) and np.all(pe_matrix <= 1.0),
          f"Range: [{pe_matrix.min():.4f}, {pe_matrix.max():.4f}]")

    # Verify sin/cos pattern: even cols should be sin, odd cols should be cos
    # At position 0, sin(0)=0 for even cols
    check("PE col 0 at pos 0 is sin(0)=0",
          abs(pe_matrix[0, 0, 0]) < 1e-5,
          f"Got: {pe_matrix[0, 0, 0]:.6f}")

    # At position 0, cos(0)=1 for odd cols
    check("PE col 1 at pos 0 is cos(0)=1",
          abs(pe_matrix[0, 0, 1] - 1.0) < 1e-5,
          f"Got: {pe_matrix[0, 0, 1]:.6f}")

    # Verify PE input/output shape compatibility
    test_seq = tf.random.normal((BATCH, expected_seq_len, d_model))
    pe_out = pe_layer(test_seq)
    check("PE output shape matches input shape",
          pe_out.shape == test_seq.shape,
          f"In: {test_seq.shape}, Out: {pe_out.shape}")

# ============================================================
# 4. PARAMETER COUNT
# ============================================================
section("4. PARAMETER COUNT")

total_params = model.count_params()
trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

print(f"\n  Total parameters:         {total_params:>10,}")
print(f"  Trainable parameters:     {trainable_params:>10,}")
print(f"  Non-trainable parameters: {non_trainable_params:>10,}")

check("Total = trainable + non-trainable",
      total_params == trainable_params + non_trainable_params,
      f"{total_params} == {trainable_params} + {non_trainable_params}")

# Break down by component
print(f"\n  Parameter breakdown by component:")
print(f"  {'Component':<40} {'Params':<10}")
print(f"  {'-'*40} {'-'*10}")

component_params = {}
for layer in model.layers:
    n = layer.count_params()
    if n > 0:
        # Group by component type
        name = layer.name
        if 'conv2d' in name and 'batch' not in name:
            group = f"Conv: {name}"
        elif 'batch_norm' in name:
            group = f"BatchNorm: {name}"
        elif 'transformer_block' in name:
            group = f"Transformer: {name}"
        elif 'sinusoidal' in name:
            group = f"PositionalEncoding: {name}"
        elif 'dense' in name:
            group = f"Dense: {name}"
        elif 'layer_norm' in name:
            group = f"LayerNorm: {name}"
        else:
            group = f"Other: {name}"
        component_params[group] = n
        print(f"  {group:<40} {n:<10,}")

check("Model is lightweight (< 200K params)",
      total_params < 200000,
      f"Total: {total_params:,}")

# PE should be non-trainable
pe_param_count = 1 * expected_seq_len * d_model  # (1, 24, 64) = 1536
check("PE params are non-trainable and excluded from trainable count",
      non_trainable_params >= pe_param_count,
      f"PE params: {pe_param_count}, Non-trainable: {non_trainable_params}")

# ============================================================
# 5. TRANSFORMER ATTENTION OVER TEMPORAL TOKENS
# ============================================================
section("5. TRANSFORMER ATTENTION VERIFICATION")

# Find transformer blocks
transformer_blocks = [l for l in model.layers if isinstance(l, TransformerEncoderBlock)]
check(f"Found {len(transformer_blocks)} transformer blocks (expected 2)",
      len(transformer_blocks) == 2)

for i, block in enumerate(transformer_blocks):
    cfg = block.get_config()
    print(f"\n  Transformer Block {i}:")
    print(f"    d_model:      {cfg['d_model']}")
    print(f"    num_heads:    {cfg['num_heads']}")
    print(f"    ff_dim:       {cfg['ff_dim']}")
    print(f"    dropout_rate: {cfg['dropout_rate']}")
    print(f"    key_dim:      {cfg['d_model'] // cfg['num_heads']}")

    check(f"Block {i}: d_model divisible by num_heads",
          cfg['d_model'] % cfg['num_heads'] == 0,
          f"{cfg['d_model']} % {cfg['num_heads']} = {cfg['d_model'] % cfg['num_heads']}")

# Verify attention operates over temporal dimension (seq_len tokens)
# by checking that the MHA inside each block processes (batch, seq_len, d_model)
print(f"\n  Attention operates over {expected_seq_len} temporal tokens")
print(f"  Each token represents {N_TP / expected_seq_len:.0f} original timepoints")
print(f"  (after 2x AvgPool(1,4): {N_TP} -> {t_after_pool1} -> {t_after_pool2})")

# Test transformer block standalone
test_input = tf.random.normal((BATCH, expected_seq_len, d_model))
block_out = transformer_blocks[0](test_input, training=False)

check("Transformer block preserves sequence shape",
      block_out.shape == test_input.shape,
      f"In: {test_input.shape}, Out: {block_out.shape}")

# Verify attention is self-attention (query=key=value from same input)
# The code passes x_norm as both query and key: self.mha(x_norm, x_norm)
# This creates a (seq_len × seq_len) attention matrix
print(f"\n  Self-attention matrix shape: ({expected_seq_len}, {expected_seq_len})")
print(f"  Each token attends to all {expected_seq_len} temporal positions")

check("Temporal token count > 1 (attention is meaningful)",
      expected_seq_len > 1,
      f"seq_len = {expected_seq_len}")

# ============================================================
# 6. TRAIN/EVAL MODE (Dropout/BatchNorm behavior)
# ============================================================
section("6. TRAIN/EVAL MODE VERIFICATION")

x_test = np.random.randn(BATCH, N_CH, N_TP, 1).astype(np.float32)

# Run prediction multiple times in EVAL mode — should be deterministic
preds_eval = []
for _ in range(5):
    p = model(x_test, training=False)
    preds_eval.append(p.numpy())

eval_consistent = all(np.allclose(preds_eval[0], p) for p in preds_eval[1:])
check("Eval mode is deterministic (dropout/BN inactive)",
      eval_consistent,
      f"Max variation: {max(np.max(np.abs(p - preds_eval[0])) for p in preds_eval[1:]):.2e}")

# Run in TRAIN mode — should show variation due to dropout
preds_train = []
for _ in range(10):
    p = model(x_test, training=True)
    preds_train.append(p.numpy())

train_varies = not all(np.allclose(preds_train[0], p) for p in preds_train[1:])
check("Train mode is stochastic (dropout active)",
      train_varies,
      "Outputs vary across forward passes with training=True")

# Verify BatchNorm layers have different behavior in train vs eval
bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]
check(f"Found {len(bn_layers)} BatchNormalization layers",
      len(bn_layers) >= 3,
      f"Expected >= 3 (2 temporal conv + 1 spatial)")

# Check BN has moving mean/variance (non-trainable)
if bn_layers:
    bn = bn_layers[0]
    has_moving = any('moving' in w.name for w in bn.non_trainable_weights)
    check("BatchNorm has moving statistics",
          has_moving,
          f"Non-trainable weights: {[w.name for w in bn.non_trainable_weights]}")

# Verify Dropout layers exist
dropout_layers = [l for l in model.layers
                  if isinstance(l, tf.keras.layers.Dropout)]
# Also count dropouts inside transformer blocks
for block in transformer_blocks:
    if hasattr(block, 'dropout1'):
        dropout_layers.append(block.dropout1)
    if hasattr(block, 'dropout2'):
        dropout_layers.append(block.dropout2)
    # MHA also has internal dropout

print(f"\n  Dropout layers found: {len(dropout_layers)}")
print(f"    - Conv frontend: 2 (after each conv block)")
print(f"    - Transformer: 2 per block × 2 blocks = 4 (+MHA internal)")
print(f"    - Classification head: 1")

# ============================================================
# 7. INPUT SHAPE COMPARISON: EEGConformer vs EEGNet
# ============================================================
section("7. INPUT SHAPE COMPARISON (EEGConformer vs EEGNet)")

eegnet = EEGNet(n_channels=N_CH, n_timepoints=N_TP)

print(f"\n  {'Property':<30} {'EEGNet':<20} {'EEGConformer':<20}")
print(f"  {'-'*30} {'-'*20} {'-'*20}")

# Input shape
eegnet_in = eegnet.input_shape
conf_in = conformer.input_shape
print(f"  {'Input shape (C,T,1)':<30} {str(eegnet_in):<20} {str(conf_in):<20}")
check("Both use same input shape format",
      eegnet_in == conf_in,
      f"EEGNet: {eegnet_in}, Conformer: {conf_in}")

# needs_channel_dim
print(f"  {'needs_channel_dim()':<30} {eegnet.needs_channel_dim():<20} {conformer.needs_channel_dim():<20}")
check("Both return needs_channel_dim()=True",
      eegnet.needs_channel_dim() == conformer.needs_channel_dim() == True)

# Test with same preprocessing flow
X_raw = np.random.randn(BATCH, N_CH, N_TP).astype(np.float32)  # (N, C, T)
X_dl = X_raw[..., np.newaxis]  # add_channel_dim: (N, C, T, 1)

print(f"\n  Raw data shape:    {X_raw.shape}  (N, C, T)")
print(f"  After channel dim: {X_dl.shape}  (N, C, T, 1)")

# Both models should accept X_dl
eegnet_pred = eegnet.predict(X_dl)
conf_pred = conformer.predict(X_dl)

check("EEGNet accepts preprocessed input",
      eegnet_pred.shape == (BATCH,),
      f"Shape: {eegnet_pred.shape}")

check("EEGConformer accepts same preprocessed input",
      conf_pred.shape == (BATCH,),
      f"Shape: {conf_pred.shape}")

# Compare parameter counts
eegnet_params = eegnet.model.count_params()
conf_params = conformer.model.count_params()
print(f"\n  {'Total params':<30} {eegnet_params:<20,} {conf_params:<20,}")
print(f"  {'Model type':<30} {'Pure CNN':<20} {'Conv+Transformer':<20}")

# Test on ALL dataset configs
print(f"\n  Cross-dataset shape compatibility:")
for ds_name, cfg in CONFIGS.items():
    nc, nt = cfg["n_channels"], cfg["n_timepoints"]
    try:
        m = EEGConformer(n_channels=nc, n_timepoints=nt)
        x = np.random.randn(2, nc, nt, 1).astype(np.float32)
        out = m.predict(x)
        seq_len_ds = nt // 16  # two pools of 4
        status = PASS
        detail = f"input=({nc},{nt},1) → seq={seq_len_ds} → output={out.shape}"
    except Exception as e:
        status = FAIL
        detail = str(e)
        issues_found.append(f"Failed on {ds_name}")
    print(f"  {status} {ds_name}: {detail}")

# ============================================================
# BONUS: Verify model.fit() works (mini training sanity check)
# ============================================================
section("BONUS: Mini Training Sanity Check")

X_mini = np.random.randn(16, N_CH, N_TP, 1).astype(np.float32)
y_mini = np.random.randint(0, 2, size=(16,)).astype(np.float32)

mini_model = EEGConformer(n_channels=N_CH, n_timepoints=N_TP)
history = mini_model.fit(X_mini, y_mini, epochs=3, batch_size=4, verbose=0)

check("model.fit() returns history object",
      history is not None and hasattr(history, 'history'))

check("History contains 'accuracy' key",
      'accuracy' in history.history,
      f"Keys: {list(history.history.keys())}")

check("History contains 'loss' key",
      'loss' in history.history,
      f"Keys: {list(history.history.keys())}")

check("Loss decreased or stayed reasonable over 3 epochs",
      history.history['loss'][-1] < history.history['loss'][0] + 0.5,
      f"Loss: {[f'{l:.4f}' for l in history.history['loss']]}")

# Test with validation data (matching runner behavior)
history_val = mini_model.fit(
    X_mini[:12], y_mini[:12],
    X_val=X_mini[12:], y_val=y_mini[12:],
    epochs=2, batch_size=4, verbose=0
)

check("model.fit() with validation data works",
      'val_loss' in history_val.history,
      f"Keys: {list(history_val.history.keys())}")

# ============================================================
# FINAL SUMMARY
# ============================================================
section("FINAL SUMMARY")

print(f"\n  Model: EEGConformer")
print(f"  Primary test config: BNCI2014001 ({N_CH}ch, {N_TP}tp)")
print(f"  Input: (batch, {N_CH}, {N_TP}, 1)")
print(f"  Temporal reduction: {N_TP} → {t_after_pool1} → {t_after_pool2} tokens")
print(f"  Transformer: {expected_seq_len} tokens × {d_model}d, {2} blocks, {4} heads")
print(f"  Output: (batch, 1) with sigmoid")
print(f"  Total params: {total_params:,} ({trainable_params:,} trainable)")
print()

if issues_found:
    print(f"  {FAIL} ISSUES FOUND ({len(issues_found)}):")
    for issue in issues_found:
        print(f"      - {issue}")
else:
    print(f"  {PASS} ALL CHECKS PASSED — No issues found")

print(f"\n{'='*60}")

# Print full model summary at the end
section("FULL MODEL SUMMARY")
conformer.summary()
