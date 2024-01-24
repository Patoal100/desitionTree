import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("Uso de espacios de parqueo").getOrCreate()

data1 = spark.read.csv("cam_1_df.csv", header=True, inferSchema=True)
data2 = spark.read.csv("cam_2_df.csv", header=True, inferSchema=True)

data1 = data1.drop(0)
data2 = data2.drop(0)

data2 = data2.fillna(1)

data1 = data1.withColumn("fecha", data1["fecha"].cast("datetime"))
data1 = data1.withColumn("hora", data1["hora"].cast("integer"))

data2 = data2.withColumn("fecha", data2["fecha"].cast("datetime"))
data2 = data2.withColumn("hora", data2["hora"].cast("integer"))

(trainingData1, testData1) = data1.randomSplit([0.8, 0.2], seed=42)
(trainingData2, testData2) = data2.randomSplit([0.8, 0.2], seed=42)

model1 = DecisionTreeClassifier(randomSeed=42)
model2 = DecisionTreeClassifier(randomSeed=42)

model1.fit(trainingData1)
model2.fit(trainingData2)

predictions1 = model1.transform(testData1)
predictions2 = model2.transform(testData2)

evaluator1 = MulticlassClassificationEvaluator(metricName="accuracy")
evaluator2 = MulticlassClassificationEvaluator(metricName="accuracy")

accuracy1 = evaluator1.evaluate(predictions1)
accuracy2 = evaluator2.evaluate(predictions2)

print(f"Precisión del modelo 1: {accuracy1}")
print(f"Precisión del modelo 2: {accuracy2}")

predictions1.toPandas().to_csv("num_unos1.csv", index=False)
predictions2.toPandas().to_csv("num_unos2.csv", index=False)


plt.bar(predictions1.select("p_").toPandas().index, predictions1.select("p_").toPandas().values)
plt.xlabel("Espacios de parqueo de la cámara 1")
plt.ylabel("Uso predecido por el modelo 1")
plt.title("Uso de espacios de parqueo de la cámara 1")
plt.savefig("grafica1.png")

plt.bar(predictions2.select("p_").toPandas().index, predictions2.select("p_").toPandas().values)
plt.xlabel("Espacios de parqueo de la cámara 2")
plt.ylabel("Uso predecido por el modelo 2")
plt.title("Uso de espacios de parqueo de la cámara 2")
plt.savefig("grafica2.png")

spark.stop()