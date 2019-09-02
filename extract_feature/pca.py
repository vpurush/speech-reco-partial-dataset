from sklearn.decomposition import PCA

pcaModels = {}
def performPCA(key, data, noOfPCAComponents = 15):
    print("data shape", data.shape)
    # for d in data:
    #     print("d.shape", d.shape)
    pca = PCA(n_components = noOfPCAComponents)
    pcaAttributes = pca.fit(data)
    pcaModels[key] = pca

    print("pcaAttributes", pcaAttributes, pcaAttributes.explained_variance_ratio_ )

    dimensionallyReducedData = pca.transform(data)
    print("dimensionallyReducedData.shape", dimensionallyReducedData.shape)

    return dimensionallyReducedData


def transformUsingPCA(key, data):
    pca = pcaModels[key]
    dimensionallyReducedData = pca.transform(data)
    print("dimensionallyReducedData.shape", dimensionallyReducedData.shape)

    return dimensionallyReducedData
