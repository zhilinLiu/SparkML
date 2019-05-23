import org.apache.spark.ml.clustering.KMeansModel
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
/*
  使用非监督学习的K-MEANS
 */

object LocalSpark {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setMaster("local[*]").setAppName("test")
    val sc = new SparkContext(conf)
    //  在线游戏时间，充值金额
    val list = List(
      (60,55),
      (90,86),
      (30,22),
      (15,11),
      (288,300),
      (0,0),
      (14,5),
      (320,280),
      (65,55),
      (13,0),
      (10,18),
      (115,108),
      (3,0),
      (52,40),
      (62,76),
      (73,80),
      (45,30),
      (1,0),
      (180,166))
    val data = sc.parallelize(list)
    val parsedData = data.map{x => Vectors.dense(x._1.toDouble,x._2.toDouble)}.cache()
    // 设置簇的个数为3
    val numClusters = 3
    // 迭代20次
    val numIterations = 20
    //  运行10次，选出最优解
    val runs = 10
    //  建模
    val clusters = KMeans.train(parsedData,numClusters,numIterations,runs)
    val WSSSE = clusters.computeCost(parsedData)
    println("WithinSet Sum of Squared Errors = " + WSSSE)
    // 根据模型预测该数据属于哪个簇
    val a1 = clusters.predict(Vectors.dense(57.0,30.0))

    //  打印测试数据属于哪个簇
    println("预测第1个用户归类为"+a1)
    parsedData.map(v=>v.toString()+"belong to cluster==="+clusters.predict(v)).foreach(println(_))
    //  保存模型
    clusters.save(sc,"/target")
    //  加载模型
    val sameModel = KMeansModel.load(sc,"/age")
    
  }
}
