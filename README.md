# zoom-sentiment-analysis

## Running API

`./bootstrap.sh`

## Input

```
{
	"text" : <text>
} 
```

## Output

```
{
	"emotion": <emotion>,
	"keywords": <list of keywords>,
	"sentiment": {
		"label": <string>,
		"score": <float>
	}
}
```