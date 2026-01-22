# Debug Summary: JAX vs PyTorch Divergence Analysis

## Key Findings

### 1. Divergence Starts Immediately
- The models diverge at **step 0** (the very first token)
- Correlation at step 0: **0.587** (very low)
- Max absolute difference at step 0: **19.18** (very high)

### 2. PyTorch Shows Repetition Behavior
- At step 0: PyTorch predicts "The" (repeating the input token 792)
- At step 2: PyTorch predicts " brown" (repeating the input token 14199)  
- At step 3: PyTorch predicts " fox" (repeating the input token 39936)
- This suggests PyTorch is getting stuck in a repetition loop

### 3. JAX Shows Reasonable Behavior
- At step 0: JAX predicts " first" (token 1177) - reasonable continuation
- At step 3: JAX predicts "es" (token 289) - reasonable continuation after "fox"

### 4. Correlation Improves Over Steps
- Step 0: 0.587 (worst)
- Step 1: 0.757
- Step 2: 0.780  
- Step 3: 0.841 (best)

### 5. Argmax Divergence Pattern
- Step 0: JAX=" first", PyTorch="The" (DIVERGENT)
- Step 1: Both=" and" (MATCH)
- Step 2: JAX=" and", PyTorch=" brown" (DIVERGENT)
- Step 3: JAX="es", PyTorch=" fox" (DIVERGENT)

## Hypothesis

The PyTorch implementation has an issue in:
1. **Initial state handling** - The first step shows the worst divergence
2. **State propagation** - PyTorch tends to repeat previous tokens
3. **Step function implementation** - May not be correctly updating states

## Next Steps for Investigation

1. **Compare initial states**: Check if JAX and PyTorch initialize states the same way
2. **Inspect step function**: Look at how `model.step()` works in both implementations
3. **Check state updates**: Verify that states are being updated correctly between steps
4. **Examine attention mechanisms**: The repetition suggests attention might not be working properly

## Technical Details

### Prompt Used
```
"The quick brown fox"
```

### Token Breakdown
- Token 0: 792 ("The")
- Token 1: 4063 (" quick")  
- Token 2: 14199 (" brown")
- Token 3: 39936 (" fox")

### Model Configuration
- Both models: 40 layers, d_model=960
- Checkpoint JAX: `seqcond-l40-d960-th15-sh30-m4-r3-o0-a0_step50000.pkl`
- Checkpoint PyTorch: `seqcond_torch_50k.pt`

## Conclusion

The PyTorch implementation has a fundamental issue that causes immediate divergence from the JAX reference implementation. The repetition behavior suggests problems with either:
- State initialization and propagation
- Attention mechanism implementation  
- Step function logic

Further investigation should focus on comparing the step-by-step state updates and attention computations between the two implementations.