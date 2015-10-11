0. литературный анализ
1. кластеризовать по средним продажам/посетителям/наличию промо/наличию праздников/среднедневным отклонениям продаж
3. сделать аккуратный мэтчинг праздников для TrainData/TestData
4. приближаем сезонность рядами Фурье?
5. предсказывать продажи конкретного магазина скользящим средним, коэффициенты подбирать для кластера


SELECT Store, MIN(DayRef), MAX(DayRef), SUM(Sales), AVG(Sales)
FROM TrainData
WHERE Sales > 0
GROUP BY Store
ORDER BY 4 desc

SELECT Promo, StateHoliday, SchoolHoliday, COUNT(*)
FROM TrainData
GROUP BY Promo, StateHoliday, SchoolHoliday


ALTER TABLE "main"."TrainData" ADD COLUMN "Promo2" int

UPDATE TrainDate
SET Promo2 = 1
FROM TrainData INNER JOIN Stores
ON TrainData.Store = Stores.Store
INNER JOIN StoresPromo
ON Stores.Store = StoresPromo.Store
WHERE Stores.Promo2 = 1
AND YEAR(TrainData.Date) >= Stores.Promo2Year
AND MONTH(TrainData.Date) = StoresPromo.Month


--Promo2WEEK !!!!!!!!!!!!!!!
