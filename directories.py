from pathlib import Path



abcdDirectory = Path.home().joinpath(r'Documents\abcd\ABCD_6')

dataDirectory = abcdDirectory.joinpath('data')
dataDirectory.mkdir(parents=True, exist_ok=True)

imagingDirectory = dataDirectory.joinpath('imaging')
imagingDirectory.mkdir(parents=True, exist_ok=True)

gordonDirectory = dataDirectory.joinpath('gordon333')
gordonDirectory.mkdir(parents=True, exist_ok=True)

figuresDirectory = abcdDirectory.joinpath('figures')
figuresDirectory.mkdir(parents=True, exist_ok=True)

modelsDirectory = abcdDirectory.joinpath('models')
modelsDirectory.mkdir(parents=True, exist_ok=True)

resultsDirectory = abcdDirectory.joinpath('results')
resultsDirectory.mkdir(parents=True, exist_ok=True)
