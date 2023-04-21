
Ngal = 300_000
pos_data = np.random.random((int(Ngal),3)) * 2000 -1000
pos_random = np.random.random((int(10*Ngal),3)) * 2000 -1000

s_edges=np.linspace(10,120,20)
bins_s = (np.array(s_edges[1:]) + np.array(s_edges[:-1])) / 2
mu_edges = np.linspace(-1,1,100)
bins_mu = (np.array(mu_edges[1:]) + np.array(mu_edges[:-1])) / 2


engine = CorrelationFunction3D(mode='smu',edges=(s_edges,mu_edges),data_positions=pos_data,random_positions=pos_random,data_weights=None,random_weights=None,RR=None,Nthread=64)
corr = engine.run()


from matplotlib import cm
fig = plt.figure(figsize=(18,9))
ax = plt.axes(projection='3d')

S,MU=np.meshgrid(bins_s,bins_mu)
S,MU = S.T,MU.T

ax.plot_surface(S,MU, corr, cmap=cm.PuOr,
                       linewidth=0, antialiased=False,alpha=0.8)

