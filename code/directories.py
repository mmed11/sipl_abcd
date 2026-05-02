from pathlib import Path

# Central directory that everything circles around
abcdDirectory = Path.home().joinpath(r'Documents\abcd\ABCD_6')
abcdDirectory.mkdir(parents=True, exist_ok=True)

dataDirectory = abcdDirectory.joinpath('data')
dataDirectory.mkdir(exist_ok=True)

imagingDirectory = dataDirectory.joinpath('imaging')
imagingDirectory.mkdir(exist_ok=True)

gordonDirectory = dataDirectory.joinpath('gordon333')
gordonDirectory.mkdir(exist_ok=True)

generalDirectory = dataDirectory.joinpath('general')
generalDirectory.mkdir(exist_ok=True)

figuresDirectory = abcdDirectory.joinpath('figures')
figuresDirectory.mkdir(exist_ok=True)

resultsDirectory = abcdDirectory.joinpath('results')
resultsDirectory.mkdir(exist_ok=True)

