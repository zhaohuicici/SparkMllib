package SparkMllibOtheralgorithm


import org.apache.spark.mllib.feature.ChiSqSelector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * 基于卡方校验的特征选择
  * 卡方校验：
  * 在分类资料统计推断中一般用于检验一个样本是否符合预期的一个分布．
  * 是统计样本的实际值与理论推断值之间的偏离程度．
  * 卡方值越小，越趋于符合
  * https://my.oschina.net/sunmin/blog/723931
  */
object SparkMllibX2FeatureSelection {
  val conf = new SparkConf()                                     //创建环境变量
    .setMaster("local")                                             //设置本地化处理
    .setAppName("TF_IDF")                              //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {

    val data = MLUtils.loadLibSVMFile(sc, "C:/Users/cici/Desktop/SparkAdvanceCourse/src/SparkMllibOtheralgorithm/fs.txt")
    val discretizedData = data.map { lp => //创建数据处理空间
      LabeledPoint(lp.label, Vectors.dense(lp.features.toArray.map {x => x/2}))
    }

    val selector = new ChiSqSelector(2)//创建选择2个特性的卡方校验
    val transformer = selector.fit(discretizedData)//创建训练模型

    val filteredData = discretizedData.map { lp =>  //过滤前两个特性
      LabeledPoint(lp.label, transformer.transform(lp.features))
    }
    filteredData.foreach(println)

    //    (0.0,[1.0,0.5])
    //    (1.0,[0.0,0.0])
    //    (0.0,[1.5,1.5])
    //    (1.0,[0.5,0.0])
    //    (1.0,[2.0,1.0])
  }
}
