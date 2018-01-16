package sticker

import (
	"sort"
	"sync"
)

var valuesOfIdealDCG = make(map[uint]float32)
var mutexValuesOfIdealDCG sync.RWMutex

// IdealDCG returns the calculated ideal DCG.
// Ideal DCG@K is defined as \sum_{k=1}^K 1/log_2(1+k), which is the maximum of possible DCG@K values.
// These value are cached persistently.
//
// Ideal DCG@0 is undefined, so this function returns NaN.
func IdealDCG(K uint) float32 {
	if K == 0 {
		return NaN32()
	}
	mutexValuesOfIdealDCG.RLock()
	I, ok := valuesOfIdealDCG[K]
	mutexValuesOfIdealDCG.RUnlock()
	if !ok {
		mutexValuesOfIdealDCG.Lock()
		for k := uint(1); k <= K; k++ {
			I += 1.0 / Log2_32(1.0+float32(k))
		}
		valuesOfIdealDCG[K] = I
		mutexValuesOfIdealDCG.Unlock()
	}
	return I
}

// InvertRanks returns the inverted ranking list.
func InvertRanks(labelRanks LabelVector) map[uint32]int {
	invRanks := make(map[uint32]int, len(labelRanks))
	for rank, label := range labelRanks {
		if label != ^uint32(0) {
			invRanks[label] = rank + 1
		}
	}
	return invRanks
}

// RankTopK returns the top-K labels.
func RankTopK(labelDist SparseVector, K uint) LabelVector {
	// When returning more than 1/10-th of the labels, if the number of the labels is the more than 25, then use the sorted labels.
	if len(labelDist) < 10*int(K) && len(labelDist) > 25 {
		label_freqs := make(KeyValues32OrderedByValue, 0, len(labelDist))
		for label, freq := range labelDist {
			label_freqs = append(label_freqs, KeyValue32{label, freq})
		}
		sort.Sort(sort.Reverse(label_freqs))
		K_ := K
		if K_ > uint(len(label_freqs)) {
			K_ = uint(len(label_freqs))
		}
		y := make(LabelVector, K)
		for i := 0; i < int(K_); i++ {
			y[i] = label_freqs[i].Key
		}
		for i := int(K_); i < len(y); i++ {
			y[i] = ^uint32(0)
		}
		return y
	}
	y, ylen := make(LabelVector, K), 0
	for label, freq := range labelDist {
		l := ylen
		for l > 0 {
			if labelDist[y[l-1]] > freq || (labelDist[y[l-1]] == freq && y[l-1] < label) {
				break
			}
			l--
		}
		if l < len(y) {
			if l < ylen {
				l_ := ylen
				if l_ >= len(y) {
					l_ = len(y) - 1
				}
				for l_ > l {
					y[l_] = y[l_-1]
					l_--
				}
			}
			y[l] = label
			if ylen < len(y) {
				ylen++
			}
		}
	}
	for l := ylen; l < int(K); l++ {
		y[l] = ^uint32(0)
	}
	return y
}

// ReportMaxPrecision reports the maximum Precision@K value of each label vector in Y.
func ReportMaxPrecision(Y LabelVectors, K uint) []float32 {
	pKs := make([]float32, len(Y))
	for i, yi := range Y {
		pK := float32(len(yi))
		if pK > float32(K) {
			pK = float32(K)
		}
		pKs[i] = pK / float32(K)
	}
	return pKs
}

// ReportNDCG reports the nDCG@K (normalized DCG@K) value of each label vector in Y.
//
// nDCG@0 is undefined, so this function returns a slice filled with NaN.
//
// NOTICE: The maximum nDCG@K is always 1.0, because nDCG@K is normalized.
func ReportNDCG(Y LabelVectors, K uint, YK_ LabelVectors) []float32 {
	pKs := make([]float32, len(Y))
	if K == 0 {
		for i, _ := range pKs {
			pKs[i] = NaN32()
		}
		return pKs
	}
	for i, yi := range Y {
		YK_i := YK_[i]
		pKi := float32(0.0)
		lenYK_i := len(YK_[i])
		if lenYK_i > int(K) {
			lenYK_i = int(K)
		}
		labelSeti := make(map[uint32]struct{})
		for _, label := range yi {
			labelSeti[label] = struct{}{}
		}
		for rank := 0; rank < lenYK_i; rank++ {
			if _, ok := labelSeti[YK_i[rank]]; ok {
				pKi += 1.0 / Log2_32(1.0+(1.0+float32(rank)))
			}
		}
		Ki := K
		if Ki > uint(len(yi)) {
			Ki = uint(len(yi))
		}
		pKs[i] = pKi / IdealDCG(Ki)
	}
	return pKs
}

// ReportPrecision reports the Precision@K value of each label vector in Y.
func ReportPrecision(Y LabelVectors, K uint, YK_ LabelVectors) []float32 {
	pKs := make([]float32, len(Y))
	for i, yi := range Y {
		YK_i := YK_[i]
		pKi := float32(0.0)
		lenYK_i := len(YK_i)
		if lenYK_i > int(K) {
			lenYK_i = int(K)
		}
		labelSeti := make(map[uint32]struct{})
		for _, label := range yi {
			labelSeti[label] = struct{}{}
		}
		for rank := 0; rank < lenYK_i; rank++ {
			if _, ok := labelSeti[YK_i[rank]]; ok {
				pKi += 1.0
			}
		}
		pKs[i] = pKi / float32(K)
	}
	return pKs
}
