reference: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

|            |             |              |        |       |        |        |           |          |
| ---------- | ----------- | ------------ | ------ | ----- | ------ | ------ | --------- | -------- |
| Fermi**†** | Kepler**†** | Maxwell**‡** | Pascal | Volta | Turing | Ampere | Lovelace* | Hopper** |
| sm_20      | sm_30       | sm_50        | sm_60  | sm_70 | sm_75  | sm_80  | *sm_*90?  | sm_100c? |
|            | sm_35       | sm_52        | sm_61  | sm_72 |        | sm_86  |           |          |
|            | sm_37       | sm_53        | sm_62  |       |        |        |           |          |

**†** Fermi and Kepler are deprecated from CUDA 9 and 11 onwards
**‡** Maxwell is deprecated from CUDA 12 onwards
\* Lovelace is the microarchitecture replacing Ampere (AD102)
** Hopper is NVIDIA’s rumored “tesla-next” series, with a 5nm process.


### Fermi cards (CUDA 3.2 until CUDA 8)

Deprecated from CUDA 9, support completely dropped from CUDA 10.

- **SM20 or SM_20, compute_30** –
  GeForce 400, 500, 600, GT-630.
  ***Completely dropped from CUDA 10 onwards.\***

### Kepler cards (CUDA 5 until CUDA 10)

Deprecated from CUDA 11.

- **SM30 or `SM_30, compute_30` –**
  Kepler architecture (e.g. generic Kepler, GeForce 700, GT-730).
  Adds support for unified memory programming
  ***Completely dropped from CUDA 11 onwards**.*
- **SM35 or `SM_35, compute_35`** –
  Tesla K40.
  Adds support for dynamic parallelism.
  **Deprecated from CUDA 11, will be dropped in future versions**.
- **SM37 or `SM_37, compute_37`** –
  Tesla K80.
  Adds a few more registers.
  ***Deprecated from CUDA 11, will be dropped in future versions***, strongly suggest replacing with a [32GB PCIe Tesla V100](https://www.amazon.com/gp/product/B07JVNHFFX/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=B07JVNHFFX&linkCode=as2&tag=arnonshimoni-20&linkId=039f38074e50b581e71d500cd08bca85).

### Maxwell cards (CUDA 6 until CUDA 11)

- **SM50 or `SM_50, compute_50`** –
  Tesla/Quadro M series.
  ***Deprecated from CUDA 11, will be dropped in future versions***, strongly suggest replacing with a [Quadro RTX 4000](https://www.amazon.com/gp/product/B07P6CDHS5/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=B07P6CDHS5&linkCode=as2&tag=arnonshimoni-20&linkId=fe1f6fa6ad408060f634a35bad4271ce).
- **SM52 or `SM_52, compute_52`** –
  Quadro M6000 , GeForce 900, GTX-970, GTX-980, GTX Titan X.
- **SM53 or `SM_53, compute_53`** –
  Tegra (Jetson) TX1 / Tegra X1, Drive CX, Drive PX, Jetson Nano.

### **Pascal (CUDA 8 and later)**

- **SM60 or `SM_60, compute_60`** –
  Quadro GP100, Tesla P100, DGX-1 (Generic Pascal)
- **SM61 or `SM_61, compute_61`**–
  GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030 (GP108), GT 1010 (GP108) Titan Xp, Tesla P40, Tesla P4, Discrete GPU on the NVIDIA Drive PX2
- **SM62 or `SM_62, compute_62`** – 
  Integrated GPU on the NVIDIA Drive PX2, Tegra (Jetson) TX2

### Volta (CUDA 9 and later)

- **SM70 or `SM_70, compute_70`** –
  DGX-1 with Volta, Tesla V100, GTX 1180 (GV104), Titan V, Quadro GV100
- **SM72 or `SM_72, compute_72`** –
  Jetson AGX Xavier, Drive AGX Pegasus, Xavier NX

### Turing (CUDA 10 and later)

- **SM75 or `SM_75, compute_75`** –
  GTX/RTX Turing – GTX 1660 Ti, RTX 2060, [RTX 2070](https://www.amazon.com/gp/product/B082P1BF7H/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=B082P1BF7H&linkCode=as2&tag=arnonshimoni-20&linkId=68e78b128dd90f652eb7796404e2126f), RTX 2080, Titan RTX, Quadro RTX 4000, Quadro RTX 5000, Quadro RTX 6000, Quadro RTX 8000, Quadro T1000/T2000, Tesla T4

### Ampere (CUDA 11.1 and later)

- **SM80 or `SM_80, compute_80`** –
  NVIDIA A100 (the name “Tesla” has been dropped – GA100), NVIDIA DGX-A100
- ***\*SM86 or `SM_86, compute_86`\** –** (from [CUDA 11.1 onwards](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html))
  Tesla GA10x cards, RTX Ampere – RTX 3080, GA102 – RTX 3090, RTX A2000, A3000, A4000, A5000, A6000, NVIDIA A40, GA106 – [RTX 3060](https://www.amazon.com/gp/product/B08W8DGK3X/ref=as_li_qf_asin_il_tl?ie=UTF8&tag=arnonshimoni-20&creative=9325&linkCode=as2&creativeASIN=B08W8DGK3X&linkId=5cb5bc6a11eb10aab6a98ad3f6c00cb9), GA104 – RTX 3070, GA107 – RTX 3050, Quadro A10, Quadro A16, Quadro A40, A2 Tensor Core GPU

> “*Devices of compute capability 8.6 have 2x more FP32 operations per cycle per SM than devices of compute capability 8.0. While a binary compiled for 8.0 will run as is on 8.6, it is recommended to compile explicitly for 8.6 to benefit from the increased FP32 throughput.*“
>
> https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#improved_fp32

### Hopper (CUDA 12 [planned] and later)

- **SM90 or `SM_90, compute_90`** –
  NVIDIA H100 (GH100)