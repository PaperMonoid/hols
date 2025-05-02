module Main where
import System.IO
import Data.List
import Control.Monad
import Text.Printf (printf)
import Control.Monad (forM_)


data LinearModel = LinearModel { getCoefficients :: [Float]
                               , getBias :: Float
                               } deriving (Show)


data Dataset = Dataset { getX :: [[Float]]
                       , getY :: [Float]
                       } deriving (Show)


data GdParams = GdParams { getIterations :: Int
                         , getDmm :: [Float]
                         , getDbm :: Float
                         } deriving (Show)


defaultGdParams :: Dataset -> GdParams
defaultGdParams dataset =
  GdParams { getIterations = 1000
           , getDmm = take (length x0) (repeat 0.0)
           , getDbm = 0.0 }
  where
    x = getX dataset
    x0 = head x


defaultLinearModel :: Dataset -> LinearModel
defaultLinearModel dataset =
  LinearModel { getCoefficients = take (length x0) (repeat 1.0)
              , getBias = 0.0 }
  where
    x = getX dataset
    x0 = head x


apply :: LinearModel -> [Float] -> Float
apply model x = b + (sum $ zipWith (*) m x)
  where m = getCoefficients model
        b = getBias model


gradientDescent :: LinearModel -> Dataset -> GdParams -> LinearModel
gradientDescent model dataset params
  | iterations <= 0 = model
  | otherwise = gradientDescent model' dataset params'
  where
    -- parameters
    gamma = 0.95
    l = 0.00001
    -- data
    m = getCoefficients model
    b = getBias model
    x = getX dataset
    y = getY dataset
    iterations = getIterations params
    dmm = getDmm params
    dbm = getDbm params
    y_hat = map (apply model) x
    diff = zipWith (-) y_hat y
    scale = 2.0 / fromIntegral (length y)
    -- gradient
    dm = [ scale * sum [ e * xij | (e, xij) <- zip diff column ]
         | column <- transpose x ]
    db = sum $ map (* scale) diff
    -- momentum
    dmm' = [ gamma * dmm_i + (1 - gamma) * dm_i
           | (dmm_i, dm_i) <- zip dmm dm ]
    dbm' = gamma * dbm + (1 - gamma) * db
    -- learning
    m' = [ m_i - l * dmm'_i
         | (m_i, dmm'_i) <- zip m dmm' ]
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
      x = take 1000 $ map takeEverythingButLast floatTable
      y = take 1000 $ concat $ map dropEverythingButLast floatTable
      dataset = Dataset { getX = x, getY = y }
      model = defaultLinearModel dataset
      params = defaultGdParams dataset
      trainedModel = gradientDescent model dataset params
      y_hat = [ apply trainedModel x_n | x_n <- x ]
  printf "Done!\n"

  printf "Printing results...\n"
  forM_ (zip y y_hat) $ \(y_n, y_hat_n) ->
    printf "%.4f  ->  %.4f\n" y_n y_hat_n
