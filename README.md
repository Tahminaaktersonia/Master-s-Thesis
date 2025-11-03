# Master-s-Thesis
ANALYSIS OF COMPRESSION TECHNIQUES FOR 5G O-RAN FRONTHAUL COMPRESSION

This MATLAB simulation, created for the Master’s Thesis “Analysis of Compression Techniques for 5G O-RAN Fronthaul Compression” (Tampere University, 2025), compares five compression techniques for 5G O-RAN IQ data: Block Floating Point (BFP), Block Scaling (BS), μ-Law Companding, Modulation-Based Compression, and Deep Learning–Based Compression. The methods are evaluated using Compression Ratio (CR) and Error Vector Magnitude (EVM) for QPSK, 16-QAM, and 64-QAM modulations across 8–14-bit quantization levels.

The output graphs show:

1. CR vs EVM: BFP and Block Scaling achieve low EVM (<10%) at moderate CR (≈2.5–3.5), μ-Law yields higher compression with increased EVM, Modulation-Based compression remains lossless (EVM ≈ 0%), and the Deep Learning method achieves the highest CR (≈6–8) with acceptable EVM.

2. CR vs Bitwidth: Compression ratio decreases as bitwidth increases for all methods, confirming the expected trade-off between compression efficiency and signal quality.

Overall, BFP and Block Scaling provide the best balance between quality and compression, μ-Law favors efficiency at some distortion, and the Deep Learning approach offers strong compression potential with moderate EVM.
