using System.Globalization;

namespace KMeans;

internal class Program
{
    private static void Main(string[] args)
    {
        Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        var data = new List<double[]>
        {
            new double[] {1, 1},
            new double[] {4, 1},
            new double[] {6, 1},
            new double[] {1, 2},
            new double[] {2, 3},
            new double[] {5, 3},
            new double[] {2, 5},
            new double[] {3, 5},
            new double[] {2, 6},
            new double[] {3, 8},
            new double[] {4, 2},
            new double[] {5, 1},
            new double[] {7, 2},
            new double[] {6, 3},
            new double[] {4, 5}
        };

        Console.WriteLine("Test");
        Console.WriteLine("Data : ");
        for (int i = 0; i < data.Count; i++)
            Console.WriteLine($"Data ke-{i + 1} : [{string.Join(", ", data[i])}]");

        var result = KMeans(data, 3, 0.2);

        Console.WriteLine("Hasil");
        Console.WriteLine($"Iterasi : {result.Iterasi}");
        Console.WriteLine($"Nilai Fungsi Objektif : {result.NilaiFungsiObjektif}");
        Console.WriteLine("Centroids : ");
        for (int i = 0; i < result.Centroids.Count; i++)
            Console.WriteLine($"Centroid Kluster ke-{i + 1} : [{string.Join(", ", result.Centroids[i])}]");
        Console.WriteLine("Clusters : ");
        for (int i = 0; i < result.Clusters.Count; i++)
            Console.WriteLine($"Data ke-{i + 1} : {result.Clusters[i] + 1}");

        //Console.WriteLine("History : ");
        //for(int k = 0; k < result.History.Count; k++)
        //{
        //    Console.WriteLine($"Iterasi ke-{k}");
        //    var (Centroids, Clusters) = result.History[k];
        //    Console.WriteLine("Centroids : ");
        //    for (int i = 0; i < Centroids.Count; i++)
        //        Console.WriteLine($"Centroid Kluster ke-{i + 1} : [{string.Join(", ", Centroids[i])}]");
        //    Console.WriteLine("Clusters : ");
        //    for (int i = 0; i < Clusters.Count; i++)
        //        Console.WriteLine($"Data ke-{i + 1} : {Clusters[i] + 1}");
        //}    
    }

    public static double Jarak(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Panjang a dan b harus sama");

        var total = 0d;

        for (var i = 0; i < a.Length; i++)
            total += Math.Pow(a[i] - b[i], 2);

        return Math.Sqrt(total);
    }

    public static double FungsiObjektif(List<double[]> centroids, List<double[]> data, List<int> clusters)
    {
        var total = 0d;

        for (var i = 0; i < data.Count; i++)
            total += Math.Pow(Jarak(centroids[clusters[i]], data[i]), 2);

        return total;
    }

    public static double[] UpdateCentroid(List<double[]> data, List<int> clusters, int indexKluster)
    {
        var dimensiData = data.Select(k => k.Length).Max();
        var centroid = new double[dimensiData];
        var jumlahData = 0;

        for (int i = 0; i < data.Count; i++)
            if (clusters[i] == indexKluster)
            {
                jumlahData++;
                for(int j = 0; j < dimensiData; j++)
                    centroid[j] += data[i][j];
            }

        for(int i = 0; i < dimensiData; i++)
            centroid[i] = centroid[i] / jumlahData;

        return centroid;
    }

    public static KMeansResult KMeans(List<double[]> data, int jumlahKluster, double thresholdFungsiObjektif = 1e-2)
    {
        if (!data.All(k => k.Length == data.Select(k => k.Length).Max()))
            throw new ArgumentException("Dimensi data harus sama");

        var random = new Random();
        var dimensiData = data[0].Length;
        var iterasi = 0;
        var nilaiFungsiObjektif = 0d;
        var nilaiFungsiObjektifLama = 0d;
        var jumlahPerpindahanCluster = data.Count;
        var centroids = Enumerable.Repeat(new double[dimensiData], jumlahKluster).ToList();
        var clusters = Enumerable.Repeat(0, data.Count).Select(i => random.Next(jumlahKluster)).ToList();
        var history = new List<(List<double[]> Centroids, List<int> Clusters, double NilaiFungsiObjektif)>();

        //Update Centroid Awal
        for (int i = 0; i < jumlahKluster; i++)
            centroids[i] = UpdateCentroid(data, clusters, i);

        //Hitung Fungsi Objektif
        nilaiFungsiObjektif = FungsiObjektif(centroids, data, clusters);

        history.Add((centroids.Select(c => c.Select(x => x).ToArray()).ToList(), clusters.Select(x => x).ToList(), nilaiFungsiObjektif));

        while(jumlahPerpindahanCluster > 0 && Math.Abs(nilaiFungsiObjektif - nilaiFungsiObjektifLama) > thresholdFungsiObjektif)
        {
            //Perbarui cluster
            jumlahPerpindahanCluster = 0;

            for(int i = 0; i < data.Count; i++)
            {
                var indexClusterBaru = -1;
                var jarakTerkecil = double.MaxValue;

                for(int j = 0; j < jumlahKluster; j++)
                {
                    var jarak = Jarak(data[i], centroids[j]);
                    if(jarak < jarakTerkecil)
                    {
                        indexClusterBaru = j;
                        jarakTerkecil = jarak;
                    }
                }

                if (indexClusterBaru != clusters[i])
                    jumlahPerpindahanCluster++;

                clusters[i] = indexClusterBaru;
            }

            //Update Centroid
            for (int i = 0; i < jumlahKluster; i++)
                centroids[i] = UpdateCentroid(data, clusters, i);

            nilaiFungsiObjektifLama = nilaiFungsiObjektif;
            nilaiFungsiObjektif = FungsiObjektif(centroids, data, clusters);
            iterasi++;

            history.Add((centroids.Select(c => c.Select(x => x).ToArray()).ToList(), clusters.Select(x => x).ToList(), nilaiFungsiObjektif));
        }

        return new KMeansResult(iterasi, nilaiFungsiObjektif, data, centroids, clusters, history);
    }
}

public sealed record KMeansResult(
    int Iterasi, 
    double NilaiFungsiObjektif,
    List<double[]> Data,
    List<double[]> Centroids,
    List<int> Clusters,
    List<(List<double[]> Centroids, List<int> Clusters, double NilaiFungsiObjektif)> History);