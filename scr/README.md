# Modifiche fatte rispetto al paper:

1. Nel paper il è CNNblock costituito d LeakyReLU -> BN -> CNN. Ora invece è **CNN -> GroupNorm -> SiLU**
2. Positional encoding 2D
3. Modifica di (CNNBlock3 + CNNBlock4): ora l'output finale è B x 512 x 3 x 6  
4. No Masked MHA nel decoder
6. modifica trainer function per gestire due GPU
7. Sequenza finale più corta: ora 18 (mean sulla direzione H), prima 108 (6*18)