
import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { API } from "@/services/api";
import MatchCard from "./MatchCard";
import { Match } from "@/types/match";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Clock, Trophy } from "lucide-react";

export default function MatchesList() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<"all" | "upcoming" | "live" | "finished">("all");
  
  const {
    data: matches,
    isLoading,
    isError,
    refetch
  } = useQuery({
    queryKey: ["matches"],
    queryFn: API.matches.getMatches
  });

  useEffect(() => {
    // Auto-refresh data every minute for live matches
    const intervalId = setInterval(() => {
      if (activeTab === "live") {
        refetch();
      }
    }, 60000); // 1 minute

    return () => clearInterval(intervalId);
  }, [activeTab, refetch]);

  const filteredMatches = matches?.filter((match) => {
    // Filter by status
    if (activeTab === "upcoming" && match.status !== "scheduled") return false;
    if (activeTab === "live" && match.status !== "live") return false;
    if (activeTab === "finished" && match.status !== "finished") return false;
    
    // Filter by search query
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return (
        match.homeTeam.name.toLowerCase().includes(query) ||
        match.awayTeam.name.toLowerCase().includes(query) ||
        match.league.name.toLowerCase().includes(query)
      );
    }
    
    return true;
  });

  // Sort matches by start time and status priority
  const sortedMatches = [...(filteredMatches || [])].sort((a, b) => {
    // Priority: 1. Live 2. Upcoming 3. Finished
    const statusPriority = { live: 0, scheduled: 1, finished: 2 };
    const statusDiff = statusPriority[a.status] - statusPriority[b.status];
    
    if (statusDiff !== 0) return statusDiff;
    
    // For same status, sort by start time
    return new Date(a.startTime).getTime() - new Date(b.startTime).getTime();
  });

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-bold mb-2">Matches</h2>
        <p className="text-gray-500">View upcoming matches and their predicted outcomes</p>
      </div>
      
      <div className="mb-6">
        <Input
          placeholder="Search teams, leagues..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="max-w-md"
        />
      </div>
      
      <Tabs defaultValue="all" onValueChange={(value) => setActiveTab(value as any)}>
        <TabsList className="mb-6">
          <TabsTrigger value="all">All Matches</TabsTrigger>
          <TabsTrigger value="upcoming" className="flex items-center gap-1">
            <Clock size={14} />
            Upcoming
          </TabsTrigger>
          <TabsTrigger value="live" className="flex items-center gap-1">
            <div className="w-2 h-2 bg-soccer-loss rounded-full animate-pulse mr-1"></div>
            Live
          </TabsTrigger>
          <TabsTrigger value="finished" className="flex items-center gap-1">
            <Trophy size={14} />
            Finished
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="all" className="mt-0">
          <MatchesGrid 
            matches={sortedMatches} 
            isLoading={isLoading} 
            isError={isError}
            searchQuery={searchQuery}
          />
        </TabsContent>
        
        <TabsContent value="upcoming" className="mt-0">
          <MatchesGrid 
            matches={sortedMatches} 
            isLoading={isLoading} 
            isError={isError}
            searchQuery={searchQuery}
          />
        </TabsContent>
        
        <TabsContent value="live" className="mt-0">
          <MatchesGrid 
            matches={sortedMatches} 
            isLoading={isLoading} 
            isError={isError}
            searchQuery={searchQuery}
          />
        </TabsContent>
        
        <TabsContent value="finished" className="mt-0">
          <MatchesGrid 
            matches={sortedMatches} 
            isLoading={isLoading} 
            isError={isError}
            searchQuery={searchQuery}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}

interface MatchesGridProps {
  matches?: Match[];
  isLoading: boolean;
  isError: boolean;
  searchQuery: string;
}

function MatchesGrid({ matches, isLoading, isError, searchQuery }: MatchesGridProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[1, 2, 3, 4, 5, 6].map((i) => (
          <div key={i} className="border rounded-lg overflow-hidden">
            <div className="h-8 bg-gray-200"></div>
            <div className="p-4 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex flex-col items-center">
                  <Skeleton className="h-12 w-12 rounded-full mb-2" />
                  <Skeleton className="h-4 w-20" />
                </div>
                <Skeleton className="h-8 w-16" />
                <div className="flex flex-col items-center">
                  <Skeleton className="h-12 w-12 rounded-full mb-2" />
                  <Skeleton className="h-4 w-20" />
                </div>
              </div>
              <div>
                <Skeleton className="h-2 w-full mt-4" />
                <div className="flex justify-between mt-2">
                  <Skeleton className="h-3 w-12" />
                  <Skeleton className="h-3 w-12" />
                  <Skeleton className="h-3 w-12" />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <div className="text-center py-12">
        <div className="text-red-500 mb-2">Failed to load matches</div>
        <button 
          onClick={() => window.location.reload()}
          className="text-soccer-green hover:underline"
        >
          Refresh page
        </button>
      </div>
    );
  }

  if (!matches?.length) {
    return (
      <div className="text-center py-12 border rounded-lg">
        <p className="text-lg font-medium">No matches found</p>
        {searchQuery && (
          <p className="text-gray-500 mt-2">
            Try changing your search query or filters
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {matches.map((match) => (
        <MatchCard key={match.id} match={match} />
      ))}
    </div>
  );
}
