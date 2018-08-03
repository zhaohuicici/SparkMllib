package SparkMllibOtheralgorithm


import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}

/**
  * 一般来说，分类是指有监督的学习，即要分类的样本是有标记的，类别是已知的；
  * 聚类是指无监督的学习，样本没有标记，根据某种相似度度量，将样本聚为　K类．
  *
  * 聚类KMEANS
  * 基本思想和核心内容就是在算法开始时随机给定若干（k）个中心，按照距离原则将样本点分配到各个中心点，
  * 之后按照平均法计算聚类集的中心点位置，从而重新确定新的中心点位置．这样不断地迭代下去直至聚类集内的样本满足一定的阈值为止．
  *
  * https://my.oschina.net/sunmin/blog/721864
  */
object SparkMllibKmeans {
  val conf = new SparkConf()                                     //创建环境变量
    .setMaster("local")                                             //设置本地化处理
    .setAppName("KMeans")                              //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
    val data = sc.textFile("./src/SparkMllibOtheralgorithm/Kmeans.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble)))
      .cache()
    val numClusters = 2 //最大分类数
    val numIterations = 20 //迭代次数
    val model = KMeans.train(parsedData, numClusters, numIterations)

    model.clusterCenters.foreach(println)//分类中心点
    //[1.4000000000000001,2.0]
    //[3.6666666666666665,3.6666666666666665]
  }
}
