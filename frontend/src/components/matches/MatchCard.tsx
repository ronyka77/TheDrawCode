
import { format, isPast, isToday } from "date-fns";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Match } from "@/types/match";

interface MatchCardProps {
  match: Match;
}

export default function MatchCard({ match }: MatchCardProps) {
  const { homeTeam, awayTeam, league, startTime, status, score, prediction } = match;
  
  const matchDate = new Date(startTime);
  const isPastMatch = isPast(matchDate);
  const isTodayMatch = isToday(matchDate);
  
  // Calculate the highest probability outcome for highlighting
  const outcomeProbabilities = [
    { type: "win", probability: prediction.homeWinProbability },
    { type: "draw", probability: prediction.drawProbability },
    { type: "loss", probability: prediction.awayWinProbability },
  ];
  
  const highestProbabilityOutcome = outcomeProbabilities.reduce((prev, current) =>
    current.probability > prev.probability ? current : prev
  );

  const renderPredictionBar = () => {
    return (
      <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
        <div className="flex h-full">
          <div 
            className="h-full bg-soccer-win" 
            style={{ width: `${prediction.homeWinProbability * 100}%` }}
          />
          <div 
            className="h-full bg-soccer-draw" 
            style={{ width: `${prediction.drawProbability * 100}%` }}
          />
          <div 
            className="h-full bg-soccer-loss" 
            style={{ width: `${prediction.awayWinProbability * 100}%` }}
          />
        </div>
      </div>
    );
  };

  const renderScoreOrTime = () => {
    if (status === "live" || status === "finished") {
      return (
        <div className="text-center">
          <div className="text-xl font-bold">
            {score?.home} - {score?.away}
          </div>
          {status === "live" && (
            <Badge variant="destructive" className="animate-pulse-light mt-1">LIVE</Badge>
          )}
        </div>
      );
    }
    
    return (
      <div className="text-center text-sm">
        <div>{format(matchDate, "MMM d")}</div>
        <div className="font-semibold">{format(matchDate, "h:mm a")}</div>
      </div>
    );
  };

  return (
    <Card className="overflow-hidden transition-shadow hover:shadow-md">
      <CardContent className="p-0">
        <div className="bg-soccer-blue text-white px-3 py-1 text-xs flex justify-between items-center">
          <div className="flex items-center">
            {league.logoUrl && (
              <img 
                src={league.logoUrl} 
                alt={league.name} 
                className="w-4 h-4 mr-1"
                onError={(e) => {
                  (e.target as HTMLImageElement).style.display = 'none';
                }}
              />
            )}
            {league.name}
          </div>
          {isTodayMatch && <Badge variant="secondary" className="text-xs">Today</Badge>}
        </div>
        
        <div className="p-4">
          <div className="flex items-center justify-between mb-4">
            {/* Home Team */}
            <div className="flex flex-col items-center max-w-[30%]">
              <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-2">
                <img 
                  src={homeTeam.logoUrl} 
                  alt={homeTeam.name}
                  className="w-8 h-8 object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = '/placeholder.svg';
                  }}
                />
              </div>
              <div className="text-center font-medium text-sm truncate w-full">
                {homeTeam.name}
              </div>
            </div>

            {/* Score / Match Time */}
            {renderScoreOrTime()}

            {/* Away Team */}
            <div className="flex flex-col items-center max-w-[30%]">
              <div className="w-12 h-12 bg-gray-100 rounded-full flex items-center justify-center mb-2">
                <img 
                  src={awayTeam.logoUrl} 
                  alt={awayTeam.name}
                  className="w-8 h-8 object-contain"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = '/placeholder.svg';
                  }}
                />
              </div>
              <div className="text-center font-medium text-sm truncate w-full">
                {awayTeam.name}
              </div>
            </div>
          </div>
          
          {prediction.status === "ready" && (
            <div className="mt-4">
              <div className="flex justify-between text-xs mb-1">
                <span>Prediction</span>
                <span className="text-gray-500 text-xs">
                  {prediction.lastUpdated && `Updated ${format(new Date(prediction.lastUpdated), "MMM d, h:mm a")}`}
                </span>
              </div>
              
              {renderPredictionBar()}
              
              <div className="flex justify-between mt-2 text-xs">
                <div className={`${highestProbabilityOutcome.type === "win" ? "font-bold" : ""}`}>
                  Home: {Math.round(prediction.homeWinProbability * 100)}%
                </div>
                <div className={`${highestProbabilityOutcome.type === "draw" ? "font-bold" : ""}`}>
                  Draw: {Math.round(prediction.drawProbability * 100)}%
                </div>
                <div className={`${highestProbabilityOutcome.type === "loss" ? "font-bold" : ""}`}>
                  Away: {Math.round(prediction.awayWinProbability * 100)}%
                </div>
              </div>
            </div>
          )}
          
          {prediction.status === "pending" && (
            <div className="mt-4 text-center py-2">
              <div className="animate-pulse flex space-x-4">
                <div className="flex-1 space-y-4 py-1">
                  <div className="h-2 bg-gray-200 rounded w-full"></div>
                  <div className="flex justify-between">
                    <div className="h-2 bg-gray-200 rounded w-1/4"></div>
                    <div className="h-2 bg-gray-200 rounded w-1/4"></div>
                    <div className="h-2 bg-gray-200 rounded w-1/4"></div>
                  </div>
                </div>
              </div>
              <p className="text-xs text-gray-500 mt-2">Prediction loading...</p>
            </div>
          )}
          
          {prediction.status === "error" && (
            <div className="mt-4 p-2 bg-red-50 border border-red-200 rounded-md text-center">
              <p className="text-xs text-red-500">Unable to load prediction</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
