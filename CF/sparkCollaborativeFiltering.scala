package SparkMllibOtheralgorithm

/**
  *  协同过滤算法，基于余弦相似度的用户相似度计算
  *  一般来说欧几里得相似度用来表现不同目标的绝对差异性，分析目标之间的相似性与差异情况．
  *  而余弦相似度更多的是对目标从前进趋势上进行区分．
  *  https://my.oschina.net/sunmin/blog/719117
  */
package spark.collaborativeFiltering

import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.Map

object sparkCollaborativeFiltering {
  val conf = new SparkConf()
    .setMaster("local")
    .setAppName("CollaborativeFilteringSpark ")	//设置环境变量
  val sc = new SparkContext(conf) //实例化环境
  val users = sc.parallelize(
    Array("张三","李四","王五","朱六","卓七")
  ) //设置用户
  val films = sc.parallelize(
    Array("飘","龙门客栈","罗密欧与朱丽叶","澳门风云","狼图腾")
  )	//设置电影名

  //使用一个source嵌套map作为姓名电影名和分值的存储
  val source = Map[String,Map[String,Int]]()
  val filmSource = Map[String,Int]()//设置一个用以存放电影分的map
  def getSource(): Map[String,Map[String,Int]] = {//设置电影评分
  val user1FilmSource = Map("飘" -> 2,"龙门客栈" -> 3,
    "罗密欧与朱丽叶" -> 1,"澳门风云" -> 0,"狼图腾" -> 1)
    val user2FilmSource = Map("飘" -> 1,"龙门客栈" -> 2,
      "罗密欧与朱丽叶" -> 2,"澳门风云" -> 1,"狼图腾" -> 4)
    val user3FilmSource = Map("飘" -> 2,"龙门客栈" -> 1,
      "罗密欧与朱丽叶" -> 0,"澳门风云" -> 1,"狼图腾" -> 4)
    val user4FilmSource = Map("飘" -> 3,"龙门客栈" -> 2,
      "罗密欧与朱丽叶" -> 0,"澳门风云" -> 5,"狼图腾" -> 3)
    val user5FilmSource = Map("飘" -> 5,"龙门客栈" -> 3,
      "罗密欧与朱丽叶" -> 1,"澳门风云" -> 1,"狼图腾" -> 2)
    source += ("张三" -> user1FilmSource)//对人名进行存储
    source += ("李四" -> user2FilmSource)
    source += ("王五" -> user3FilmSource)
    source += ("朱六" -> user4FilmSource)
    source += ("卓七" -> user5FilmSource)
    source			//返回嵌套map
  }

  //两两计算分值,采用余弦相似性
  def getCollaborateSource(user1:String,user2:String):Double = {
    val user1FilmSource = source.get(user1)
      .get.values.toVector	//获得第1个用户的评分
    val user2FilmSource = source.get(user2)
      .get.values.toVector	//获得第2个用户的评分
    val member = user1FilmSource.zip(user2FilmSource)
      .map(d => d._1 * d._2).reduce(_ + _)
      .toDouble//对公式分子部分进行计算,zip将若干RDD 压缩成一个RDD
    val temp1  = math.sqrt(user1FilmSource.map(num => {	//求出分母第1个变量值
      math.pow(num,2)	//数学计算
    }).reduce(_ + _))	//进行叠加
    val temp2  = math.sqrt(user2FilmSource.map(num => {//求出分母第2个变量值
      math.pow(num,2)//数学计算
    }).reduce(_ + _))//进行叠加
    val denominator = temp1 * temp2	//求出分母
    member / denominator//进行计算
  }

  def main(args: Array[String]) {
    getSource()		//初始化分数
    val name = "李四"    //设定目标对象
    users.foreach(user =>{//迭代进行计算
      println(name + " 相对于 " + user +"的相似性分数是："+
        getCollaborateSource(name,user))
    })
  }
}
