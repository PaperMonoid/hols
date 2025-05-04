module Main where
import Control.Monad
import Text.Printf (printf)
import qualified Data.Vector as V


type FloatMatrix = V.Vector (V.Vector Float)

type FloatVector = V.Vector Float

data LinearModel = LinearModel { getCoefficients :: FloatVector
                               , getBias :: Float }

data Dataset = Dataset { getX :: FloatMatrix
                       , getXt :: FloatMatrix
                       , getY :: FloatVector
                       }

data GdParams = GdParams { getIterations :: Int
                         , getDmm :: FloatVector
                         , getDbm :: Float
                         }


defaultGdParams :: Dataset -> GdParams
defaultGdParams dataset =
  GdParams { getIterations = 1000
           , getDmm = V.replicate (V.length x0) 0.0
           , getDbm = 0.0 }
  where
    x0 = V.head $ getX dataset


defaultLinearModel :: Dataset -> LinearModel
defaultLinearModel dataset =
  LinearModel { getCoefficients = V.replicate (V.length x0) 1.0
              , getBias = 0.0 }
  where
    x0 = V.head $ getX dataset


dot :: FloatVector -> FloatVector -> Float
dot v w = V.sum (V.zipWith (*) v w)


apply :: LinearModel -> FloatVector -> Float
apply model x = b + (dot m x)
  where m = getCoefficients model
        b = getBias model


vtranspose :: FloatMatrix -> FloatMatrix
vtranspose rows
  | V.null rows = V.empty
  | otherwise =
    let nRows = V.length rows
        nCols = V.length (V.unsafeHead rows)
    in  V.generate nCols $ \j ->
      V.generate nRows $ \i ->
      V.unsafeIndex (V.unsafeIndex rows i) j


gradientDescent :: LinearModel -> Dataset -> GdParams -> LinearModel
gradientDescent model dataset params
  | iterations <= 0 = model
  | otherwise = gradientDescent model' dataset params'
  where
    -- parameters
    gamma = 0.95
    l = 0.00001
    -- data
    LinearModel { getCoefficients = m, getBias = b } = model
    Dataset { getX = x, getXt = x_t, getY = y } = dataset
    GdParams { getIterations = iterations, getDmm = dmm, getDbm = dbm } = params
    y_hat = V.map (apply model) x
    diff = V.zipWith (-) y_hat y
    scale = 2.0 / fromIntegral (V.length y)
    -- gradient
    dm = V.map (\col -> scale * (dot diff col)) x_t
    db = V.sum $ V.map (* scale) diff
    -- momentum
    dmm' = V.zipWith (+)
           (V.map (* gamma) dmm)
           (V.map (* (1 - gamma)) dm)
    dbm' = gamma * dbm + (1 - gamma) * db
    -- learning
    m' = V.zipWith (-) m (V.map (* l) dmm')
    b' = b - l * dbm'
    model' = LinearModel { getCoefficients = m', getBias = b' }
    params' = GdParams { getIterations = (iterations - 1)
                       , getDmm = dmm',
                         getDbm = dbm' }


splitBy :: Char -> String -> [String]
splitBy separator text =
  case break (== separator) text of
    (field, []) -> [field]
    (field, _:rest) -> field : splitBy separator rest


stringToFloat :: String -> Float
stringToFloat text = read text :: Float


takeEverythingButLast :: [a] -> [a]
takeEverythingButLast xs = take ((length xs) - 1) xs


dropEverythingButLast :: [a] -> [a]
dropEverythingButLast xs = drop ((length xs) - 1) xs


main :: IO ()
main = do
  printf "About to read file datasets/MNIST.csv...\n"
  contents <- readFile "datasets/MNIST.csv"
  printf "Done!\n"

  printf "Starting training process...\n"
  let rows = lines contents
      table = map (splitBy ',') rows
      floatTable = map (map stringToFloat) table
      x = V.fromList
        $ take 1000
        $ map (V.fromList . takeEverythingButLast) floatTable
      y = V.fromList
        $ take 1000
        $ concat
        $ map dropEverythingButLast floatTable
      dataset = Dataset { getX = x, getXt = vtranspose x, getY = y }
      model = defaultLinearModel dataset
      params = defaultGdParams dataset
      trainedModel = gradientDescent model dataset params
      y_hat = V.map (apply trainedModel) x
  printf "Done!\n"

  printf "Printing results...\n"
  forM_ (zip (V.toList y) (V.toList y_hat)) $ \(y_n, y_hat_n) ->
    printf "%.4f  ->  %.4f\n" y_n y_hat_n
