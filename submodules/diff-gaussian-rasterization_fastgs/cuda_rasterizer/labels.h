#ifndef CUDA_RASTERIZER_LABELS_H_INCLUDED
#define CUDA_RASTERIZER_LABELS_H_INCLUDED

// Optional BEA-GS label inputs/outputs. All pointers null = stock FastGS.
//   votes[g * L + l] += alpha * T   for every pixel with label l (label voting)
//   sag[p]           = sum alpha * T over Gaussians whose label != pixel label
struct LabelArgs
{
	const int* gaussian_labels = nullptr;  // [P]
	const int* pixel_labels = nullptr;     // [H * W], -1 = unlabeled pixel
	int L = 0;                             // number of labels
	float* sag = nullptr;                  // [H * W]
	float* votes = nullptr;                // [P * L]
	const float* dL_dsag = nullptr;        // [H * W], backward only
};

#endif
