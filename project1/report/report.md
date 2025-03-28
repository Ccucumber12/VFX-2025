# Project 1: High Dynamic Range Imaging

- B10902064 張晴昀
- B10902067 黃允謙

## 1. Description of this project

This project involves assembling an HDR image from a series of photographs taken at different exposures. HDR images capture a wider dynamic range and maintain a linear relationship with physical irradiance, making them useful for graphics and vision applications.

## 2. Processing flow

1. Image alignment (Bonus)

2. HDR construction

   - Debevec's method

   - Robertson's method (Bonus)

3. Tone mapping (Bonus)

## 3. Image alignment



## 4. HDR construction

We implemented two ways to compute responsive curves:

### Debevec's method

Reference: http://www.csie.ntu.edu.tw/~cyy/courses/vfx/papers/Debevec1997RHD.pdf

According to this paper, we wanted to minimize $$O = \sum\limits_{i = 1}^{N}\sum\limits_{j = 1}^{P} \{w(Z_{ij})[g(Z_{ij}) - \ln{E_i} - \ln{\Delta t_j}]\}^2 + \lambda \sum\limits_{z = Z_{min} + 1}^{Z_{max} - 1} [w(z)g''(z)]^2$$.

We first selected sample pixels and obtained $g$ curves for each color channel independently by transforming the objective function into a least-squares problem and solving it.

### Robertson's method

Reference: https://www.csie.ntu.edu.tw/~cyy/courses/vfx/papers/Robertson2003ETA.pdf

This method is aimed at obtaining $\hat{g}, \hat{E_i} = \arg \min\limits_{g, E_i} \sum\limits_{ij} w(Z_{ij})(g(Z_{ij}) - E_i \Delta t_j)^2$.

The process is repeated iteratively until convergence:

- Assume $g(Z_{ij})$ is known:

  Solve for $E_i$ using the equation $E_i = \frac{\sum\limits_{j} w(Z_{ij})g(Z_{ij})\Delta t_j}{\sum\limits_{j} w(Z_{ij})\Delta t_j^2}$.

- Assume $E_i$ is known:

  Solve for $g$ using the equation $g(m) = \frac{1}{|E_m|} \sum\limits_{ij\in E_m} E_i \Delta t_j$

  and then nomorlize $g$ such that $g(128) = 0$.

  (The initial guess for $g$ is chosen as a linear function.)

### Radiance map

After obtaining $g$ for each channel, the next step is to construct the HDR radiance map.

We computed $\ln E_i = \frac{\sum_{j = 1}^{P} w(Z_{ij})(g(Z_{ij}) - \ln \Delta t_j)}{\sum_{j = 1}^{P} w(Z_{ij})}$ for each pixel and we can then recover the radiance $E_i$ for the entire image by exponentiating the result.



## 5. Tone mapping



## 6. Comparison

### Original images

- Folder: `/data/cosmology-hall`

| 1/256![148A8373](../data/ldr/cosmology-hall/148A8373.JPG)   | 1/128![148A8374](../data/ldr/cosmology-hall/148A8374.JPG)   | 1/64![148A8376](../data/ldr/cosmology-hall/148A8376.JPG)    | 1/32![148A8380](../data/ldr/cosmology-hall/148A8380.JPG)  | 1/16![148A8382](../data/ldr/cosmology-hall/148A8382.JPG)  |
| ----------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| **1/8**![148A8384](../data/ldr/cosmology-hall/148A8384.JPG) | **1/4**![148A8386](../data/ldr/cosmology-hall/148A8386.JPG) | **1/2**![148A8388](../data/ldr/cosmology-hall/148A8388.JPG) | **1**![148A8389](../data/ldr/cosmology-hall/148A8389.JPG) | **2**![148A8391](../data/ldr/cosmology-hall/148A8391.JPG) |

### Result

- Folder:
