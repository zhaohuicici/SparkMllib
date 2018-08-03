package SparkMllibOtheralgorithm



import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.{SparkContext, SparkConf}

/**
  * TF-IDF  是一种简单的文本特征提取算法
  *  词频(Term Frequency): 某个关键词在文本中出现的次数
  *  逆文档频率(Inverse Document Frequency): 大小与一个词的常见程度成反比
  *  TF = 某个词在文章中出现的次数/文章的总词数
  *  IDF = log(查找的文章总数　/ (包含该词的文章数　+ 1))
  *  TF-IDF = TF(词频)　x IDF(逆文档频率)
  *  此处未考虑去除停用词(辅助词，如副词，介词等)和
  *  语义重构("数据挖掘"，＂数据结构＂，拆分成＂数据＂，＂挖掘＂，＂数据＂，＂结构＂)
  *  这样两个完全不同的文本具有　50% 的相似性，是非常严重的错误．
  * https://my.oschina.net/sunmin/blog/723929
  */
object SparkMllibTF_IDF {
  val conf = new SparkConf()               //创建环境变量
    .setMaster("local")                                             //设置本地化处理
    .setAppName("TF_IDF")                              //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
    val documents = sc.textFile("C:/Users/cici/Desktop/SparkAdvanceCourse/src/SparkMllibOtheralgorithm/a.txt")
      .map(_.split(" ").toSeq)

    val hashingTF = new HashingTF()			//首先创建TF计算实例
    val tf = hashingTF.transform(documents).cache()//计算文档TF值
    val idf = new IDF().fit(tf)						//创建IDF实例并计算

    val tf_idf = idf.transform(tf) //计算TF_IDF词频
    tf_idf.foreach(println)

    //    (1048576,[179334,596178],[1.0986122886681098,0.6931471805599453])
    //    (1048576,[586461],[0.1823215567939546])
    //    (1048576,[422129,586461],[0.6931471805599453,0.1823215567939546])
    //    (1048576,[586461,596178],[0.1823215567939546,0.6931471805599453])
    //    (1048576,[422129,586461],[0.6931471805599453,0.1823215567939546])
  }
}
