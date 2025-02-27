### Some useful scripts

#### rsync

Pull everything from `directory/to/results` into current directory,
excluding directories that match "*_seed*"
```shell
rsync -zLurP --exclude "*_seed*/" src:"directory/to/results" ./
```

### Parsing
