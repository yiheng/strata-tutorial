# strata-tutorial
A tutorial repository for Strata Beijing 2016. It contains setup instructions and example code.

# 环境设置
1. 安装Git
2. 使用Git将本项目clone到本地
3. 安装最新的JDK，设置JAVA_HOME环境变量
4. 安装最新的maven
5. 安装Apache Spark, 设置HADOOP_HOME环境变量（注意：在Windows环境下需要把HADOOP_HOME环境变量设为项目中hadoop\win64所在的路径），运行<code>spark-shell</code>检查Spark是否装好（注意：Windows下Saprk退出时会有无法删除临时目录的Exception）。
6. 使用maven构建项目代码，即运行<code>mvn pakcage</code>
7. 运行第一个示例，即运行<code>spark-submit --class Pipeline target/strata-tutorial-1.0-SNAPSHOT-jar-with-dependencies.jar dataset\adult.data dataset\adult.test</code>
8. 运行第二个示例，即运行<code>spark-submit --class GridSearch target/strata-tutorial-1.0-SNAPSHOT-jar-with-dependencies.jar dataset\adult.data dataset\adult.test</code>
9. 运行第三个示例，即运行<code>spark-submit --class Persist target/strata-tutorial-1.0-SNAPSHOT-jar-with-dependencies.jar dataset\adult.data dataset\adult.test</code>
10. 运行MLlib KMeans，即运行<code>spark-submit --master "local[4]" --driver-memory 4g --class com.intel.sparseml.example.KMeanTest target/strata-tutorial-1.0-SNAPSHOT.jar 100 1e6 2e6 1e-5 10 mllib</code>
11. 运行SparseKMeans，即运行<code>spark-submit --master "local[4]" --driver-memory 4g --class com.intel.sparseml.example.KMeanTest target/strata-tutorial-1.0-SNAPSHOT.jar 100 1e6 2e6 1e-5 10 sparseKMeans</code>


有问题可以给yiheng.wang@intel.com发邮件。
