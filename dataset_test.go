package sticker

import (
	"bytes"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/hiro4bbh/go-assert"
)

func TestFloat32Slice(t *testing.T) {
	x := []float32{5.0, 3.0, 1.0, 4.0, 2.0}
	sort.Sort(Float32Slice(x))
	goassert.New(t, []float32{1.0, 2.0, 3.0, 4.0, 5.0}).Equal(x)
}

func TestSummarizeFloat32Slice(t *testing.T) {
	min0, q25_0, med0, q75_0, max0, mean0 := SummarizeFloat32Slice([]float32{})
	goassert.New(t, "NaN,NaN,NaN,NaN,NaN,NaN").Equal(fmt.Sprintf("%f,%f,%f,%f,%f,%f", min0, q25_0, med0, q75_0, max0, mean0))
	goassert.New(t, float32(1.0), float32(1.0), float32(1.0), float32(1.0), float32(1.0), float32(1.0)).Equal(SummarizeFloat32Slice([]float32{1.0}))
	goassert.New(t, float32(1.0), float32(1.25), float32(1.5), float32(1.75), float32(2.0), float32(1.5)).Equal(SummarizeFloat32Slice([]float32{1.0, 2.0}))
	goassert.New(t, float32(1.0), float32(2.0), float32(3.0), float32(4.0), float32(5.0), float32(3.0)).Equal(SummarizeFloat32Slice([]float32{5.0, 3.0, 1.0, 4.0, 2.0}))
}

func TestAvgTotalVariationAmongSparseVectors(t *testing.T) {
	goassert.New(t, float32(0.0)).Equal(AvgTotalVariationAmongSparseVectors(SparseVectors{}))
	goassert.New(t, "NaN").Equal(fmt.Sprintf("%g", AvgTotalVariationAmongSparseVectors(SparseVectors{
		SparseVector{0: 1.0, 1: 1.0, 2: 1.0},
		SparseVector{},
	})))
	goassert.New(t, float32(0.0)).Equal(AvgTotalVariationAmongSparseVectors(SparseVectors{
		SparseVector{0: 1.0, 1: 1.0, 2: 1.0},
	}))
	goassert.New(t, float32(0.0)).Equal(AvgTotalVariationAmongSparseVectors(SparseVectors{
		SparseVector{0: 1.0, 1: 1.0, 2: 1.0},
		SparseVector{0: 1.0, 1: 1.0, 2: 1.0},
	}))
	goassert.New(t, float32(0.25)).Equal(AvgTotalVariationAmongSparseVectors(SparseVectors{
		SparseVector{0: 1.0, 1: 1.0, 2: 1.0, 4: 1.0},
		SparseVector{0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
	}))
	goassert.New(t, float32(0.0)).Equal(AvgTotalVariationAmongSparseVectors(createBenchmarkAvgTotalVariationAmongSparseVectors()))
}

func TestSparsifyVector(t *testing.T) {
	goassert.New(t, SparseVector{
		1: 1.0, 3: -1.0,
	}).Equal(SparsifyVector([]float32{0.0, 1.0, 0.0, -1.0, 0.0}))
}

func TestKeyValues32OrderedByKey(t *testing.T) {
	data := KeyValues32OrderedByKey(KeyValues32{
		KeyValue32{4, 2.0}, KeyValue32{1, 1.0}, KeyValue32{1, 2.0}, KeyValue32{2, 3.0}, KeyValue32{3, 3.0},
	})
	sort.Sort(data)
	goassert.New(t, KeyValues32{
		KeyValue32{1, 1.0}, KeyValue32{1, 2.0}, KeyValue32{2, 3.0}, KeyValue32{3, 3.0}, KeyValue32{4, 2.0},
	}).Equal(KeyValues32(data))
}

func TestKeyValues32OrderedByValue(t *testing.T) {
	data := KeyValues32OrderedByValue([]KeyValue32{
		KeyValue32{5, 2.0}, KeyValue32{1, 1.0}, KeyValue32{2, 1.0}, KeyValue32{3, 1.0}, KeyValue32{4, 3.0},
	})
	sort.Sort(data)
	goassert.New(t, KeyValues32{
		KeyValue32{3, 1.0}, KeyValue32{2, 1.0}, KeyValue32{1, 1.0}, KeyValue32{5, 2.0}, KeyValue32{4, 3.0},
	}).Equal(KeyValues32(data))
}

func TestFeatureVectors(t *testing.T) {
	X := FeatureVectors{
		FeatureVector{KeyValue32{0, 1.0}},
		FeatureVector{KeyValue32{0, 2.0}, KeyValue32{1, 3.0}},
		FeatureVector{KeyValue32{0, 4.0}, KeyValue32{2, 5.0}, KeyValue32{9, 6.0}},
	}
	goassert.New(t, 10).Equal(X.Dim())
}

func TestLabelVector(t *testing.T) {
	y := LabelVector{3, 2, 1}
	sort.Sort(y)
	goassert.New(t, LabelVector{1, 2, 3}).Equal(y)
}

func TestLabelVectors(t *testing.T) {
	Y := LabelVectors{
		LabelVector{0}, LabelVector{0, 1}, LabelVector{0, 2, 7},
	}
	goassert.New(t, 8).Equal(Y.Dim())
}

func TestDatasetFeatureSubSet(t *testing.T) {
	ds := &Dataset{
		X: FeatureVectors{
			FeatureVector{KeyValue32{0, 1.0}},
			FeatureVector{KeyValue32{0, 2.0}, KeyValue32{1, 3.0}},
			FeatureVector{KeyValue32{0, 4.0}, KeyValue32{2, 5.0}, KeyValue32{9, 6.0}},
			FeatureVector{KeyValue32{10, 7.0}},
		},
		Y: LabelVectors{
			LabelVector{0}, LabelVector{0, 1}, LabelVector{0, 2, 9}, LabelVector{10},
		},
	}
	subds := &Dataset{
		X: FeatureVectors{
			FeatureVector{KeyValue32{0, 1.0}},
			FeatureVector{KeyValue32{0, 2.0}},
			FeatureVector{KeyValue32{0, 4.0}, KeyValue32{9, 6.0}},
			FeatureVector{},
		},
		Y: LabelVectors{
			LabelVector{0}, LabelVector{0, 1}, LabelVector{0, 2, 9}, LabelVector{10},
		},
	}
	goassert.New(t, subds).Equal(ds.FeatureSubSet(map[uint32]struct{}{0: struct{}{}, 5: struct{}{}, 9: struct{}{}}))
}

func TestDatasetSize(t *testing.T) {
	ds := &Dataset{
		X: FeatureVectors{
			FeatureVector{KeyValue32{0, 1.0}},
			FeatureVector{KeyValue32{0, 2.0}, KeyValue32{1, 3.0}},
			FeatureVector{KeyValue32{0, 4.0}, KeyValue32{2, 5.0}, KeyValue32{9, 6.0}},
		},
		Y: LabelVectors{
			LabelVector{0}, LabelVector{0, 1}, LabelVector{0, 2, 9}, LabelVector{10},
		},
	}
	goassert.New(t, 3).Equal(ds.Size())
}

func TestDatasetSubSet(t *testing.T) {
	ds := &Dataset{
		X: FeatureVectors{
			FeatureVector{KeyValue32{0, 1.0}},
			FeatureVector{KeyValue32{0, 2.0}, KeyValue32{1, 3.0}},
			FeatureVector{KeyValue32{0, 4.0}, KeyValue32{2, 5.0}, KeyValue32{9, 6.0}},
		},
		Y: LabelVectors{
			LabelVector{0}, LabelVector{0, 1}, LabelVector{0, 2, 9}, LabelVector{10},
		},
	}
	goassert.New(t, &Dataset{
		X: FeatureVectors{ds.X[1], ds.X[1], ds.X[0]},
		Y: LabelVectors{ds.Y[1], ds.Y[1], ds.Y[0]},
	}).Equal(ds.SubSet([]int{1, 1, 0}))
}

func TestDatasetWriteToText(t *testing.T) {
	ds := &Dataset{
		X: FeatureVectors{
			FeatureVector{KeyValue32{0, 1.0}},
			FeatureVector{KeyValue32{0, 2.0}, KeyValue32{1, 3.0}},
			FeatureVector{KeyValue32{0, 4.0}, KeyValue32{2, 5.0}, KeyValue32{9, 6.0}},
			FeatureVector{KeyValue32{10, 7.0}},
		},
		Y: LabelVectors{
			LabelVector{0}, LabelVector{0, 1}, LabelVector{0, 2, 9}, LabelVector{10},
		},
	}
	var buf bytes.Buffer
	goassert.New(t).SucceedWithoutError(ds.WriteTextDataset(&buf))
	ds2 := goassert.New(t).SucceedNew(ReadTextDataset(&buf)).(*Dataset)
	goassert.New(t, ds).Equal(ds2)
}

func TestReadTextDataset(t *testing.T) {
	ds := &Dataset{
		X: FeatureVectors{
			FeatureVector{KeyValue32{0, 1.0}},
			FeatureVector{KeyValue32{0, 2.0}, KeyValue32{1, 3.0}},
			FeatureVector{KeyValue32{0, 4.0}, KeyValue32{2, 5.0}, KeyValue32{9, 6.0}},
		},
		Y: LabelVectors{
			LabelVector{0}, LabelVector{0, 1}, LabelVector{0, 2, 9},
		},
	}
	goassert.New(t, ds).EqualWithoutError(ReadTextDataset(strings.NewReader("3 10 10\r\n0 0:1\n0,1 0:2 1:3\r\n0,2,9 0:4 2:5 9:6\n")))
	goassert.New(t, "cannot read first line").ExpectError(ReadTextDataset(strings.NewReader("")))
	goassert.New(t, "illegal first line").ExpectError(ReadTextDataset(strings.NewReader("3\n")))
	goassert.New(t, "illegal nentries in first line").ExpectError(ReadTextDataset(strings.NewReader("x y z\n")))
	goassert.New(t, "illegal nfeatures in first line").ExpectError(ReadTextDataset(strings.NewReader("3 y z\n")))
	goassert.New(t, "illegal nlabels in first line").ExpectError(ReadTextDataset(strings.NewReader("3 10 z\n")))
	goassert.New(t, "L1: cannot read line").ExpectError(ReadTextDataset(strings.NewReader("3 10 10\n")))
	goassert.New(t, "L1: illegal #1 label ID").ExpectError(ReadTextDataset(strings.NewReader("3 10 10\nx\n")))
	goassert.New(t, "L1: too large #1 label ID \\(>= 10\\)").ExpectError(ReadTextDataset(strings.NewReader("3 10 10\n10\n")))
	goassert.New(t, "L1: illegal #1 featureID:value pair").ExpectError(ReadTextDataset(strings.NewReader("3 10 10\n0 x\n")))
	goassert.New(t, "L1: illegal #1 feature ID").ExpectError(ReadTextDataset(strings.NewReader("3 10 10\n0 x:y\n")))
	goassert.New(t, "L1: too large #1 feature ID \\(>= 10\\)").ExpectError(ReadTextDataset(strings.NewReader("3 10 10\n0 10:y\n")))
	goassert.New(t, "L1: illegal #1 feature value").ExpectError(ReadTextDataset(strings.NewReader("3 10 10\n0 0:y\n")))
}

func createBenchmarkAvgTotalVariationAmongSparseVectors() SparseVectors {
	n, m := 50, 20
	svs := make(SparseVectors, n)
	for i, _ := range svs {
		sv := make(SparseVector, m)
		for j := 0; j < m; j++ {
			sv[uint32(j)] = float32(j)
		}
		svs[i] = sv
	}
	return svs
}

func BenchmarkAvgTotalVariationAmongSparseVectors(b *testing.B) {
	svs := createBenchmarkAvgTotalVariationAmongSparseVectors()
	b.ResetTimer()
	for t := 0; t < b.N; t++ {
		AvgTotalVariationAmongSparseVectors(svs)
	}
}
