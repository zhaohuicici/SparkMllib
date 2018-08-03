package SparkMllibOtheralgorithm


import org.apache.spark.mllib.clustering.{PowerIterationClustering}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 快速迭代聚类
  * 基本原理：使用含有权重的无向线将样本数据连接在一张无向图中，之后按照相似度划分，
  * 使得划分后的子图内部具有最大的相似度二不同的子图具有最小的相似度从而达到聚类的效果．
  * 数据源要求　　RDD[(Long), (Long), (Double)]
  * 第一个参数和第二个参数是第一个点和第二个点的编号，即其之间 ID，第三个参数是相似度计算值．
  * https://my.oschina.net/sunmin/blog/723350
  */
object SparkMllibCluster {
  val conf = new SparkConf()                                     //创建环境变量
    .setMaster("local")                                             //设置本地化处理
    .setAppName("pic")                              //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
    val data = sc.textFile("./src/SparkMllibOtheralgorithm/pic.txt")
    val similarities = data.map { line =>
      val parts = line.split(" ")
      (parts(0).toLong, parts(1).toLong, parts(2).toDouble)
    }
    val pic = new PowerIterationClustering()
      .setK(2) //设置聚类数
      .setMaxIterations(10) //设置迭代次数
    val model = pic.run(similarities)

    model.assignments.foreach {a =>
      println(s"${a.id} -> ${a.cluster}")
    }
  }
}
