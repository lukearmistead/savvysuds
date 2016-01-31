
#########################################################
###       The Most FT Beers In Your Inventory         ###
#########################################################
-- Of all the beers you listed in your inventory, this will list them in order
-- of most often offered For Trade by BEX users to least often offered For
-- Trade by BEX users.

-- TOGGLES
SET @NUM_DAYS = 200;
SET @FTISO = "ft";

SELECT
  store_items.id,
  store_shop.name,
  store_items.beer_id,
  beers.name,
  count(ftiso.beer_id),
  ftiso.created
FROM store_items
-- join to filter brews
LEFT OUTER JOIN ftiso
  ON store_items.beer_id = ftiso.beer_id
-- join to get beer names
LEFT OUTER JOIN beers
  ON beers.id = store_items.beer_id
  -- join to get shop name
LEFT OUTER JOIN store_shop
  ON store_items.shop_id = store_shop.id
WHERE ftiso.type = @FTISO
  AND DATE(ftiso.created) > (NOW() - INTERVAL @NUM_DAYS DAY)

  -- AND DATE_SUB(NOW() - ftiso.created) < INTERVAL 30 DAY
GROUP BY shop_id, store_items.beer_id
ORDER BY shop_id, count(ftiso.beer_id) DESC;

#########################################################
###          FT breweries in your inventory           ###
#########################################################
-- Of all the beers you listed in your inventory, this will list them in order
-- of most often offered For Trade by BEX users to least often offered For
-- Trade by BEX users.

-- TOGGLES
SET @NUM_DAYS = 200;
SET @FTISO = "ft";

SELECT
  shop_id,
  store_shop.name,
  beers.brewery_id,
  count(ftiso.beer_id) AS popularity
FROM store_items
-- join to filter brews
LEFT OUTER JOIN ftiso
  ON store_items.beer_id = ftiso.beer_id
-- join to get shop name
LEFT OUTER JOIN store_shop
  ON store_items.shop_id = store_shop.id
-- joins to find brewery info
LEFT OUTER JOIN beers
  ON beers.id = store_items.beer_id
LEFT OUTER JOIN breweries
  ON beers.brewery_id = breweries.id
WHERE ftiso.type = @FTISO
  AND DATE(ftiso.created) > (NOW() - INTERVAL @NUM_DAYS DAY)
GROUP BY shop_id, beers.brewery_id
ORDER BY shop_id, count(ftiso.beer_id) DESC;


#########################################################
### Popular nearby breweries you don't carry ###
#########################################################
-- Of all the breweries you listed in your inventory, this will list them in
-- order of most often offered For Trade by BEX users to least often offered
-- For Trade by BEX users.

--482

-- TOGGLES

-- breweries held in shop inventory
SELECT
  store_items.shop_id,
  store_shop.name,
  all_breweries.brewery_id,
  all_breweries.brewery_name,
  all_breweries.brewery_state,
  all_breweries.popularity
FROM store_items

-- for conditional formatting below
LEFT OUTER JOIN beers
  ON store_items.beer_id = beers.id
LEFT OUTER JOIN breweries
  ON breweries.id = beers.brewery_id
LEFT OUTER JOIN store_shop
  ON store_items.shop_id = store_shop.id

-- popularity of beers
LEFT OUTER JOIN (
  SELECT
    breweries.id AS brewery_id,
    breweries.name AS brewery_name,
    breweries.state AS brewery_state,
    count(ftiso.beer_id) AS popularity
  FROM breweries
  LEFT OUTER JOIN beers
    ON breweries.id = beers.brewery_id
  LEFT OUTER JOIN ftiso
    ON beers.id = ftiso.beer_id
  WHERE ftiso.type = 'ft'
    AND DATE(ftiso.created) > (NOW() - INTERVAL 200 DAY)
  GROUP BY breweries.id
) AS all_breweries

-- conditionals
ON store_shop.state = all_breweries.brewery_state COLLATE utf8_unicode_ci
WHERE breweries.id <> all_breweries.brewery_id
GROUP BY all_breweries.brewery_id
ORDER BY store_items.shop_id, all_breweries.popularity DESC;
