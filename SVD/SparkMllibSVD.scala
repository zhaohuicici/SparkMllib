package SparkMllibOtheralgorithm



import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.{SparkContext, SparkConf}

/**
  * 数据降维
  * 一个矩阵在计算过程中，将它在一个方向上进行拉伸，需要关心的是拉伸的幅度与方向．
  * 奇异值分解(SVD)：一个矩阵分解成带有方向向量的矩阵相乘
  * https://my.oschina.net/sunmin/blog/723853
  */
object SparkMllibSVD{
  val conf = new SparkConf()                                     //创建环境变量
    .setMaster("local")                                             //设置本地化处理
    .setAppName("SVD")                              //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
    val data = sc.textFile("./src/SparkMllibOtheralgorithm/aaa.txt")
      .map(_.split(" ").map(_.toDouble))
      .map(line => Vectors.dense(line))

    val rm = new RowMatrix(data)                       //读入行矩阵
    val SVD = rm.computeSVD(2, computeU = true)			 //进行SVD计算
    println(SVD)			 //打印SVD结果矩阵
    //求　SVD 分解的矩阵
    println("*********************")
    val u = SVD.U
    val s = SVD.s
    val v = SVD.V

    println(u, s, v)
  }
}
