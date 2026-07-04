
## Experiment 1 - 3 component ablation - Wasserstein eval:
### Setup:
* MLP : inter_dim=256
* Mesh grid : n_theta = 20 n_phi = 40
* FM steps : 4
* Epochs: 1000
* L_max = 5
* alpha = -1 beta=2.0 nu=3.0 tau=2.0
    
### Results
* Standard FM :  25.4032436717684
* ours 3 enabled : 8.138891474830453
* ours , matern, Loss : 7.982494539658626
* ours , only matern : 26.143762084462043
* ours , only loss : 25.453320618984804
### Questions : 
*  we see the great impact of sampling fro a matern with the loss, if we consider the source distribution as prior belief on the target , how can pick the best prior , and would an idea where the source gets updated through put training via bayesian update be of an interest?

## Experiment 2 - L_max impact - Standard vs ablation winner : 
### Setup:
* Same mesh / FM / hyperparams as Exp 1 except **Epochs: 500**
* Sweep L_max = 2 .. 19
* Compare: Standard FM vs ours (Matérn source + spectral loss)
### Results
* Plot: compare_wasserstein_runs.png (script: compare_wasserstein_runs.py)
* Standard FM ~25.6--26.6 (flat across L_max)
* Ours ~7.1--9.7; minimum W2 ≈ 7.065 at L_max = 4
## Experiment 3 - intermediate FM wasserstein to see convergence : 
* 
## Experiment 4 - alpha impact - plot : 
* 
## Experiment 5 - spectral decay and harmonic analyis : 
* 
## Experiment 6 - posterior inference
## Experiment 7 - anisotropic dataset 
## Experiment 8 - Neural operator
### setup 
* modes = 5 (both)
* intermediate 
### Results 
* FNO : we had 9 , 11. , 8.4 best = 7.4 
* Spectral basis : 7.846819755666608, way too long computation
deeper network , less epochs : 8.022323802256519
more fourier modes 24 modes same 100 epochs : 7.648777327815233
 48 modes same 100 epochs less layers (1) and width 64 :7.83877371806792 way faster though
128 modes :  7.435017844745024
* Laplacian NORM
laplacian_fno Wasserstein: 12.5137
* Sobolev NO 
sobolev_fno Wasserstein: 11.2163
### setup
fno_modes=256,
fno_width=32,
fno_layers=2,
### Results
fno2d: 10.2507
  laplacian_fno: 11.5675
  sobolev_fno: 10.3993
I feel one of the biggest issues here is that we are using the 1D conv , which loses strucutral dependancy in lapl and sobolev

## Experiment 9 - Neural operator with fourier basis

## Experiment 10 - AD3 molecular dataset? conformer generation from MD run
metrics = benchmark_molecule_ad3(
        use_sobolev=False,
        model_type="mlp",
        num_epochs=1000,
        dataset_size=8192,
        batch_size=512,
        test_samples=1024,
        num_steps=8,
    )
    Euclidean Wasserstein distance: 0.6708
Sobolev spectral Wasserstein distance (beta=2.0): 0.5223
COM-aligned RMSD (nm): mean=0.1264, std=0.0178
Smoke benchmark metrics: {'w2_euclidean': np.float64(0.6708393185834334), 'w2_sobolev': 0.522331520094018, 'rmsd': {'mean': 0.12640848410211236, 'std': 0.017798449497247713}}

now for use_sobolev false:
Euclidean Wasserstein distance: 0.8676
COM-aligned RMSD (nm): mean=0.1720, std=0.0312
Smoke benchmark metrics: {'w2_euclidean': np.float64(0.8676472134172597), 'rmsd': {'mean': 0.17201188790131494, 'std': 0.031180079703114934}}

other exp :
mlp: W2=0.8151, RMSD=0.1455, Sobolev W2=0.6615
  fno1d: W2=0.7530, RMSD=0.1143, Sobolev W2=0.6243
  graph_laplacian_fno: W2=0.9439, RMSD=0.1592, Sobolev W2=0.7506
  graph_sobolev_fno: W2=1.2015, RMSD=0.1849, Sobolev W2=0.9863
## Experimet ? - impact of FM steps

## Experiment - standard s2 spherical harmonics signal 
Use_sobolev TRUE  
  Epoch 500 
    mlp: 7.2079
    fno2d: 13.9168
    laplacian_fno: 7.6228
    sobolev_fno: 29.1491
  Epochs 1000
    mlp: 7.6333
    fno2d: 7.6448
    laplacian_fno: 7.3687
    sobolev_fno: 32.4413
Use_sobolev False
  mlp: 28.3925
  fno2d: 6.3565
  laplacian_fno: 7.0223
  sobolev_fno: 6.3950
## expeirment dipeptide conformer generation - sobolev method with matern
 mlp: W2=0.8151, RMSD=0.1455, Sobolev W2=0.6615
  fno1d: W2=0.7530, RMSD=0.1143, Sobolev W2=0.6243
  graph_laplacian_fno: W2=0.9439, RMSD=0.1592, Sobolev W2=0.7506
  graph_sobolev_fno: W2=1.2015, RMSD=0.1849, Sobolev W2=0.9863
## experiment dipeptide conformer generation - standards fm 
  mlp: W2=1.0152, RMSD=0.2055
  fno1d: W2=1.3400, RMSD=0.1671
  graph_laplacian_fno: W2=1.3816, RMSD=0.1875
  graph_sobolev_fno: W2=1.3362, RMSD=0.1772
## experiment - dipeptide - make laplacian heat kernel graph
mlp: W2=1.2609, RMSD=0.2226, Sobolev W2=0.1408
  fno1d: W2=0.9134, RMSD=0.1311, Sobolev W2=0.3762
  graph_laplacian_fno: W2=1.7469, RMSD=0.1682, Sobolev W2=0.1960
  graph_sobolev_fno: W2=1.6251, RMSD=0.2218, Sobolev W2=0.1674


## comparaison heat vs bond graph laplacian 
setup : num_epochs=1000,
        dataset_size=16384,
        batch_size=1024,
        test_samples=2048,
        num_steps=8,
results: 

Heat : 

  mlp: W2=1.2436, RMSD=0.2363, Sobolev W2=0.1383
  fno1d: W2=1.0811, RMSD=0.1174, Sobolev W2=0.3761
  graph_laplacian_fno: W2=1.2120, RMSD=0.2030, Sobolev W2=0.1143
  graph_sobolev_fno: W2=1.2336, RMSD=0.1887, Sobolev W2=0.1169

Bond:
   mlp: W2=0.7236, RMSD=0.1390, Sobolev W2=0.5672
  fno1d: W2=0.7043, RMSD=0.1068, Sobolev W2=0.5802
  graph_laplacian_fno: W2=1.1213, RMSD=0.1970, Sobolev W2=0.8765
  graph_sobolev_fno: W2=0.9485, RMSD=0.1506, Sobolev W2=0.7638

Standard FM (bond for the fnos): 
  mlp: W2=0.8977, RMSD=0.1835
  fno1d: W2=1.2700, RMSD=0.1551
  graph_laplacian_fno: W2=1.3848, RMSD=0.1926
  graph_sobolev_fno: W2=1.3635, RMSD=0.1859

Standard FM (heat for the fnos):
  mlp: W2=0.9034, RMSD=0.1847
  fno1d: W2=1.2921, RMSD=0.1580
  graph_laplacian_fno: W2=1.2933, RMSD=0.1725
  graph_sobolev_fno: W2=1.2815, RMSD=0.1676