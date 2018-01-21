package sticker

import (
	"bufio"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// Float32Slice implements the interface sort.Interface.
type Float32Slice []float32

func (x Float32Slice) Len() int {
	return len(x)
}

func (x Float32Slice) Less(i, j int) bool {
	return x[i] < x[j]
}

func (x Float32Slice) Swap(i, j int) {
	x[i], x[j] = x[j], x[i]
}

// SummarizeFloat32Slice returns the 5-summary(minimum, 1st quantile, median, 3rd quantile and maximum) and the average of the given float32 slice.
func SummarizeFloat32Slice(x []float32) (min, q25, med, q75, max, avg float32) {
	if len(x) == 0 {
		return NaN32(), NaN32(), NaN32(), NaN32(), NaN32(), NaN32()
	} else if len(x) == 1 {
		return x[0], x[0], x[0], x[0], x[0], x[0]
	}
	xs := make([]float32, len(x))
	copy(xs, x)
	sort.Sort(Float32Slice(xs))
	min, max = xs[0], xs[len(xs)-1]
	if len(xs)%2 == 1 {
		med = xs[len(xs)/2]
	} else {
		med = 0.5*xs[len(xs)/2-1] + 0.5*xs[len(xs)/2]
	}
	r25 := (1.0-0.25)*0.0 + 0.25*float32(len(xs)-1)
	if _, r25f := Modf32(r25); r25f == 0.0 {
		q25 = xs[int(Floor32(r25))]
	} else {
		q25 = (Ceil32(r25)-r25)*xs[int(Floor32(r25))] + (r25-Floor32(r25))*xs[int(Ceil32(r25))]
	}
	r75 := (1.0-0.75)*0.0 + 0.75*float32(len(xs)-1)
	if _, r75f := Modf32(r75); r75f == 0.0 {
		q75 = xs[int(Floor32(r75))]
	} else {
		q75 = (Ceil32(r75)-r75)*xs[int(Floor32(r75))] + (r75-Floor32(r75))*xs[int(Ceil32(r75))]
	}
	for _, x := range xs {
		avg += x
	}
	avg /= float32(len(xs))
	return
}

// SparseVector is the map from uint32 key to float32 value.
type SparseVector map[uint32]float32

// SparseVectors is the slice of SparseVector.
type SparseVectors []SparseVector

// AvgTotalVariationAmongSparseVectors returns the average total-variation distance among the sparse vectors.
// This function returns 0.0 if there is at most one sparse vector, and returns NaN if some sparse vectors are empty.
func AvgTotalVariationAmongSparseVectors(svs SparseVectors) float32 {
	n := len(svs)
	if n <= 1 {
		return 0.0
	}
	keySet := make(map[uint32]bool)
	Zs := make([]float32, n)
	for i, svi := range svs {
		Z := float32(0.0)
		for key, value := range svi {
			keySet[key] = true
			Z += value
		}
		if Z == 0 {
			return NaN32()
		}
		Zs[i] = Z
	}
	keys := make([]int, 0, len(keySet))
	for key := range keySet {
		keys = append(keys, int(key))
	}
	sort.Ints(keys)
	sum := float32(0.0)
	for i := 0; i < n; i++ {
		svi, Zi := svs[i], Zs[i]
		for j := i + 1; j < n; j++ {
			svj, Zj := svs[j], Zs[j]
			for _, key := range keys {
				sum += Abs32(svi[uint32(key)]/Zi - svj[uint32(key)]/Zj)
			}
		}
	}
	return 0.5 * sum / (float32(n) * float32(n-1) / 2)
}

// SparsifyVector returns a SparseVector converted from v.
func SparsifyVector(v []float32) SparseVector {
	sv := make(map[uint32]float32)
	for i, vi := range v {
		if vi != 0.0 {
			sv[uint32(i)] = vi
		}
	}
	return sv
}

// KeyValue32 is the pair of uint32 feature key and its float32 value.
type KeyValue32 struct {
	Key   uint32
	Value float32
}

// KeyValues32 is the KeyValue32 slice.
type KeyValues32 []KeyValue32

// KeyValues32OrderedByKey is KeyValues32 implementing sort.Interface for sorting in the key order.
// If the keys are same, then these key-values are sorted in increasing value order.
type KeyValues32OrderedByKey KeyValues32

func (kvs KeyValues32OrderedByKey) Len() int {
	return len(kvs)
}

func (kvs KeyValues32OrderedByKey) Less(i, j int) bool {
	return kvs[i].Key < kvs[j].Key || (kvs[i].Key == kvs[j].Key && kvs[i].Value < kvs[j].Value)
}

func (kvs KeyValues32OrderedByKey) Swap(i, j int) {
	kvs[i], kvs[j] = kvs[j], kvs[i]
}

// KeyValues32OrderedByValue is KeyValues32 implementing sort.Interface for sorting in the value order.
// If the values are same, then these key-values are sorted in decreasing key order, because this is intended for sorting in decreasing key/value order in reverse mode.
type KeyValues32OrderedByValue KeyValues32

func (kvs KeyValues32OrderedByValue) Len() int {
	return len(kvs)
}

func (kvs KeyValues32OrderedByValue) Less(i, j int) bool {
	return kvs[i].Value < kvs[j].Value || (kvs[i].Value == kvs[j].Value && kvs[i].Key >= kvs[j].Key)
}

func (kvs KeyValues32OrderedByValue) Swap(i, j int) {
	kvs[i], kvs[j] = kvs[j], kvs[i]
}

// FeatureVector is the sparse and static feature vector.
// The elements should be ordered by feature ID (key).
type FeatureVector = KeyValues32OrderedByKey

// FeatureVectors is the FeatureVector slice.
type FeatureVectors []FeatureVector

// Dim returns the calculated dimension of FeatureVectors.
// This is the maximum feature key plus 1.
func (X FeatureVectors) Dim() (d int) {
	for _, xi := range X {
		for _, xipair := range xi {
			if d <= int(xipair.Key) {
				d = int(xipair.Key) + 1
			}
		}
	}
	return
}

// LabelVector is the sparse label vector which is the slice of label key.
type LabelVector []uint32

func (labels LabelVector) Len() int {
	return len(labels)
}

func (labels LabelVector) Less(i, j int) bool {
	return labels[i] < labels[j]
}

func (labels LabelVector) Swap(i, j int) {
	labels[i], labels[j] = labels[j], labels[i]
}

// LabelVectors is the LabelVector slice.
type LabelVectors []LabelVector

// Dim returns the calculated dimension of label vectors.
// This is the maximum label ID plus 1.
func (Y LabelVectors) Dim() (d int) {
	for _, yi := range Y {
		for _, label := range yi {
			if d <= int(label) {
				d = int(label) + 1
			}
		}
	}
	return
}

// Dataset is a collection of the pair of one feature vector and one label vector.
type Dataset struct {
	X FeatureVectors
	Y LabelVectors
}

// ReadTextDataset returns a new Dataset from reader.
//
// The data from reader is assumed to be formatted like used in LIBLINEAR and LIBSVM.
// It is a plain text whose cells are separated by single space.
// The first line has the number of entries, features, and labels.
// The remaining lines have entries of the dataset.
// Each entry is encoded in one line like (comma-separated label) (feature:value)*.
//
// This function returns an error in reading the dataset.
func ReadTextDataset(reader io.Reader) (*Dataset, error) {
	br := bufio.NewReader(reader)
	line, err := br.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("cannot read first line")
	}
	line = strings.TrimSpace(line)
	nentries_nfeatures_nlabels := strings.Split(line, " ")
	if len(nentries_nfeatures_nlabels) != 3 {
		return nil, fmt.Errorf("illegal first line")
	}
	nentries, err := strconv.ParseUint(nentries_nfeatures_nlabels[0], 10, 32)
	if err != nil {
		return nil, fmt.Errorf("illegal nentries in first line")
	}
	nfeatures, err := strconv.ParseUint(nentries_nfeatures_nlabels[1], 10, 32)
	if err != nil {
		return nil, fmt.Errorf("illegal nfeatures in first line")
	}
	nlabels, err := strconv.ParseUint(nentries_nfeatures_nlabels[2], 10, 32)
	if err != nil {
		return nil, fmt.Errorf("illegal nlabels in first line")
	}
	X := make(FeatureVectors, nentries)
	Y := make(LabelVectors, nentries)
	for i := uint64(0); i < nentries; i++ {
		line, err := br.ReadString('\n')
		if err != nil {
			return nil, fmt.Errorf("L%d: cannot read line", i+1)
		}
		line = strings.TrimSpace(line)
		entry := strings.Split(line, " ")
		labelStrs := strings.Split(entry[0], ",")
		entry = entry[1:]
		y := make(LabelVector, 0, len(labelStrs))
		for j, labelStr := range labelStrs {
			label, err := strconv.ParseUint(labelStr, 10, 32)
			if err != nil {
				return nil, fmt.Errorf("L%d: illegal #%d label ID", i+1, j+1)
			}
			if label >= nlabels {
				return nil, fmt.Errorf("L%d: too large #%d label ID (>= %d)", i+1, j+1, nlabels)
			}
			y = append(y, uint32(label))
		}
		x := make(FeatureVector, 0, len(entry))
		for j, feature_valueStr := range entry {
			feature_value := strings.Split(feature_valueStr, ":")
			if len(feature_value) != 2 {
				return nil, fmt.Errorf("L%d: illegal #%d featureID:value pair", i+1, j+1)
			}
			feature, err := strconv.ParseUint(feature_value[0], 10, 32)
			if err != nil {
				return nil, fmt.Errorf("L%d: illegal #%d feature ID", i+1, j+1)
			}
			if feature >= nfeatures {
				return nil, fmt.Errorf("L%d: too large #%d feature ID (>= %d)", i+1, j+1, nfeatures)
			}
			value, err := strconv.ParseFloat(feature_value[1], 32)
			if err != nil {
				return nil, fmt.Errorf("L%d: illegal #%d feature value", i+1, j+1)
			}
			x = append(x, KeyValue32{uint32(feature), float32(value)})
		}
		sort.Sort(x)
		X[i], Y[i] = x, y
	}
	return &Dataset{
		X: X,
		Y: Y,
	}, nil
}

// FeatureSubSet returns the sub-set of the dataset whose entry has only features in the given set of features.
// For efficiency, the label vectors of sub-dataset has references to the one of the dataset.
func (ds *Dataset) FeatureSubSet(features map[uint32]struct{}) *Dataset {
	subds := &Dataset{
		X: FeatureVectors{},
		Y: LabelVectors{},
	}
	for i, xi := range ds.X {
		xi_, yi_ := FeatureVector{}, ds.Y[i]
		for _, xipair := range xi {
			if _, ok := features[xipair.Key]; ok {
				xi_ = append(xi_, xipair)
			}
		}
		subds.X, subds.Y = append(subds.X, xi_), append(subds.Y, yi_)
	}
	return subds
}

// Size returns the size of the dataset.
func (ds *Dataset) Size() int {
	return len(ds.X)
}

// SubSet returns the sub-set of the dataset selected by the given index slice.
// For efficiency, the all vectors of sub-dataset has references to the ones of the dataset.
//
// This function panics if the index is out of range.
func (ds *Dataset) SubSet(indices []int) *Dataset {
	subds := &Dataset{
		X: make(FeatureVectors, len(indices)),
		Y: make(LabelVectors, len(indices)),
	}
	for ii, i := range indices {
		subds.X[ii], subds.Y[ii] = ds.X[i], ds.Y[i]
	}
	return subds
}

// WriteTextDataset writes the dataset to the given writer as ReadTextDataset supports.
//
// This function returns an error in writing.
func (ds *Dataset) WriteTextDataset(w io.Writer) error {
	if _, err := fmt.Fprintf(w, "%d %d %d\n", ds.Size(), ds.X.Dim(), ds.Y.Dim()); err != nil {
		return err
	}
	for i, xi := range ds.X {
		for j, label := range ds.Y[i] {
			if j > 0 {
				if _, err := w.Write([]byte(",")); err != nil {
					return err
				}
			}
			if _, err := fmt.Fprintf(w, "%d", label); err != nil {
				return err
			}
		}
		for _, xipair := range xi {
			if _, err := fmt.Fprintf(w, " %d:%f", xipair.Key, xipair.Value); err != nil {
				return err
			}
		}
		if _, err := w.Write([]byte("\n")); err != nil {
			return err
		}
	}
	return nil
}
