package SparkMllibOtheralgorithm

import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}

/**
  * 关联规则：研究不同类型的物品相互之间的关联关系的规则．
  * 应用于＂超市购物分析＂( 啤酒与尿布), ＂网络入侵检测＂，＂医学病例共同特征挖掘＂
  * 支持度：表示 X　和　Y　中的项在同一条件下出现的次数
  * 置信度：表示 X 和　Y　中的项在一定条件下出现的概率
  * Apriori算法：属于候选消除算法．是一个生成候选集，消除不满足条件的候选集，不断循环，知道不在产生候选集的过程．
  * FP-growth算法过程：
  * (1) 扫描样本数据库，将样本按照体递减规则排序，删除小于最小支持度的样本数
  * (2) 重新扫描样本数据库，并将样本按照上标的支持度数据排列
  * (3) 将重新生成的表按顺序插入 FP 树中,继续生成FP树，直到形成完整的FP树
  * (4) 建立频繁项集规则　
  * FP-Growth
  * https://my.oschina.net/sunmin/blog/723852
  *
  */
object SparkMllibFp_growth {
  val conf = new SparkConf()                                     //创建环境变量
    .setMaster("local")                                             //设置本地化处理
    .setAppName("fp-growth")                              //设定名称
  val sc = new SparkContext(conf)

  def main(args: Array[String]) {
    val data = sc.textFile("./src/SparkMllibOtheralgorithm/fp.txt")
    val transactions: RDD[Array[String]] = data.map(s => s.trim.split(' '))
    val fp = new FPGrowth()
      .setMinSupport(0.5)//设置最小支持度与整体的比值
      .setNumPartitions(10)//设置分区数

    val model = fp.run(transactions)

    model.freqItemsets.collect().foreach { itemset =>
      println(itemset.items.mkString("[", ",", "]") + ", " + itemset.freq)
    }

    //      [z], 5
    //      [x], 4
    //      [x,z], 3
    //      [y], 3
    //      [y,x], 3
    //      [y,x,z], 3
    //      [y,z], 3
    //      [r], 3
    //      [s], 3
    //      [s,x], 3
    //      [t], 3
    //      [t,y], 3
    //      [t,y,x], 3
    //      [t,y,x,z], 3
    //      [t,y,z], 3
    //      [t,x], 3
    //      [t,x,z], 3
    //      [t,z], 3


  }
}
