// NDWI using sentinel 2 


/**
 * Function to mask clouds using the Sentinel-2 QA band
 * @param {ee.Image} image Sentinel-2 image
 * @return {ee.Image} cloud masked Sentinel-2 image
 */
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}

var dataset = ee.ImageCollection('COPERNICUS/S2')
                  .filterDate('2021-03-05', '2021-03-10')
                  .filterBounds(piura)
                  // Pre-filter to get less cloudy granules.
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',10))
                  .map(maskS2clouds)
                  
var image = dataset.median().clip(piura);

var visualization = {
  min: 0,
  max: 0.3,
  bands: ['B4', 'B3', 'B2'],
};
// get date of image
// print("Sentinel image taken at = ", image.date());

// add layer to map
Map.centerObject(piura, 10);
Map.addLayer(image, visualization, 'RGB');

// compute NDWI/MNDWI
// var ndwi_nir = image.normalizedDifference(['B3', 'B8']).rename('NDWI') // (Green - NIR / Green + NIR)
var ndwi_swir = image.normalizedDifference(['B3', 'B12']).rename('NDWI SWIR') // (Green - SWIR / Green + SWIR)

// add layer to map
Map.addLayer(ndwi_swir, {palette: ['red', 'yellow', 'green', 'cyan', 'blue']}, 'NDWI');

// Create NDWI mask
var ndwiThreshold = ndwi_swir.gte(0.4);
var ndwiMask = ndwiThreshold.updateMask(ndwiThreshold);
Map.addLayer(ndwiThreshold, {palette:['black', 'white']}, 'NDWI Binary Mask');
Map.addLayer(ndwiMask, {palette:['blue']}, 'NDWI Mask');

  
  
   
// Get the previous 5 years of permanent water.

// Include JRC layer on surface water seasonality to mask flood pixels from areas
// of "permanent" water (where there is water > 10 months of the year)
var swater = ee.Image('JRC/GSW1_0/GlobalSurfaceWater').select('seasonality');
var swater_mask = swater.gte(10).updateMask(swater.gte(10));

 //Flooded layer where perennial water bodies (water > 10 mo/yr) is assigned a 0 value
var flooded_mask = ndwiMask.where(swater_mask,0);
// final flooded area without pixels in perennial waterbodies
var flooded = flooded_mask.updateMask(flooded_mask);
      
// Compute connectivity of pixels to eliminate those connected to 8 or fewer neighbours
// This operation reduces noise of the flood extent product 
var connections = flooded.connectedPixelCount();    
var flooded = flooded.updateMask(connections.gte(8));
      
// Mask out areas with more than 5 percent slope using a Digital Elevation Model 
var DEM = ee.Image('WWF/HydroSHEDS/03VFDEM');
var terrain = ee.Algorithms.Terrain(DEM);
var slope = terrain.select('slope');
var flooded = flooded.updateMask(slope.lt(5));

// add layer to map
Map.addLayer(flooded, {palette:['blue']}, 'Flooded Areas');

// Calculate flood extent area
// Create a raster layer containing the area information of each pixel 
var flood_pixelarea = flooded.multiply(ee.Image.pixelArea());

// Sum the areas of flooded pixels
// default is set to 'bestEffort: true' in order to reduce compuation time, for a more 
// accurate result set bestEffort to false and increase 'maxPixels'. 
var flood_stats = flood_pixelarea.reduceRegion({
  reducer: ee.Reducer.sum(),              
  geometry: piura,
  scale: 10, // native resolution 
  maxPixels: 1e13,
  bestEffort: true
  });
// print(flood_stats) // 39179456.81496498m2

// Convert the flood extent to hectares (area calculations are originally given in meters)  
var flood_area_ha = flood_stats.getNumber('NDWI SWIR').divide(10000).round(); 

// print(flood_area_ha) // 3918 ha

 // -------------------- DAMAGE ASSESSMENT ---------------------- //
  // ASSESSING POPULATION AFFECTED

// get world population in 2015 from GHSL
// recent pop data should be used here for better results
var pop = ee.ImageCollection("JRC/GHSL/P2016/POP_GPW_GLOBE_V1");
 
// Extract the projection before doing any computation
var projection = ee.Image(pop.first()).projection()
// print('Native Resolution:', projection.nominalScale()) 
// 250 meters
 
// Filter and get image for the year of interest
var filtered = pop.filter(ee.Filter.date('2015-01-01', '2016-01-01'))
var pop2015 = ee.Image(filtered.first())

// Reproject flood layer to GHSL scale
var flooded_reprj = flooded
    .reproject({
    crs: projection
  });

// Create a raster showing exposed population only using the resampled flood layer
var population_exposed = pop2015.updateMask(flooded_reprj).updateMask(pop2015);

// Calculate Total Population exposed to floods
var stats = population_exposed.reduceRegion({
  reducer: ee.Reducer.sum(),
  geometry: piura.geometry(),
  scale: projection.nominalScale(),
})
// Result is a dictionary
// Extract the value and print it
print(stats.get('population_count'))

//-------------------------------------------------//
// ASSESSING AGRICULTURAL LAND AFFECTED

// classify the sentinel image to 4 classes urban, agricultural, water and grassland

// first compute ndvi and add it for prediction
var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
var img = image.addBands(ndvi)

// add layer to map
// Map.addLayer(img, {bands: ['ndvi'], min: -1, max:1, palette: ['#654321','#FFA500','#FFFF00', '#00FF00', '#008000']}, 'ndvi') 

// Use these bands for prediction.
var bands = ['B2', 'B3', 'B4', 'B8', 'NDVI'];

var points = urban.merge(water).merge(cropland).merge(grassland)

// Get the values for all pixels in each polygon in the training.
var training = img.select(bands).sampleRegions({
  // Get the sample from the polygons FeatureCollection.
  collection: points,
  // Keep this list of properties from the polygons.
  properties: ['class'],
  // Set the scale to get S2 pixels in the polygons.
  scale: 30
});

// create an SVM classifier with custom params.
// var classifier = ee.Classifier.libsvm({
//   kernelType: 'RBF',
//   gamma: 0.5,
//   cost: 10
// });

// create a RF classifier with custom params.
var classifier = ee.Classifier.smileRandomForest(50).train({
  features: training,  
  classProperty: 'class', 
  inputProperties: bands
});

// Train the classifiers.
var trained = classifier.train(training, 'class', bands);

// Classify the image.
var classified = img.select(bands).classify(trained).rename('classes');

// Display the classification result and the input image.
Map.addLayer(classified,
            {min: 0, max: 3, palette: ['grey', 'blue', 'green', 'yellow']},
            'Classified');

// get area for cropland classs from classified image
var cropland = classified.select('classes').eq(2);//cropland has 2 value in this case

// Calculate affected cropland using the flood layer
var cropland_affected = flooded.updateMask(cropland)

//Calculate the pixel area in ha
var area_cropland = cropland_affected.multiply(ee.Image.pixelArea()).divide(100*100);

//Reducing the statistics for your study area
var stat = area_cropland.reduceRegion ({
  reducer: ee.Reducer.sum(),
  geometry: piura,
  scale: 10,
  maxPixels: 1e13
});

//Get the cropland area affected in hectares
print ('Cropland Area (in ha)', stat); //877.9450334333924ha