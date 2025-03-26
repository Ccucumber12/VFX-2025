# project #1: High Dynamic Range Imaging

- B10902064 張晴昀
- B10902067 黃允謙

## 1. Description of this project

This project involves assembling an HDR image from a series of photographs taken at different exposures. HDR images capture a wider dynamic range and maintain a linear relationship with physical irradiance, making them useful for graphics and vision applications.

## 2. Processing flow

1. Image alignment

2. HDR construction

   - Debevec's method

   - Robertson's method

3. Tone mapping

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



## 6. Result

### Camera Setting

