
import { Match, User } from '@/types/match';

// Mock data for development
const MOCK_MATCHES: Match[] = [
  {
    id: '1',
    homeTeam: {
      id: 'team1',
      name: 'Arsenal',
      logoUrl: '/placeholder.svg',
      code: 'ARS'
    },
    awayTeam: {
      id: 'team2',
      name: 'Manchester City',
      logoUrl: '/placeholder.svg',
      code: 'MCI'
    },
    league: {
      id: 'league1',
      name: 'Premier League',
      country: 'England',
      logoUrl: '/placeholder.svg'
    },
    startTime: new Date(Date.now() + 3600000).toISOString(),
    status: 'scheduled',
    prediction: {
      homeWinProbability: 0.35,
      drawProbability: 0.25,
      awayWinProbability: 0.40,
      status: 'ready',
      lastUpdated: new Date().toISOString()
    }
  },
  {
    id: '2',
    homeTeam: {
      id: 'team3',
      name: 'Barcelona',
      logoUrl: '/placeholder.svg',
      code: 'BAR'
    },
    awayTeam: {
      id: 'team4',
      name: 'Real Madrid',
      logoUrl: '/placeholder.svg',
      code: 'RMD'
    },
    league: {
      id: 'league2',
      name: 'La Liga',
      country: 'Spain',
      logoUrl: '/placeholder.svg'
    },
    startTime: new Date(Date.now() + 7200000).toISOString(),
    status: 'scheduled',
    prediction: {
      homeWinProbability: 0.45,
      drawProbability: 0.30,
      awayWinProbability: 0.25,
      status: 'ready',
      lastUpdated: new Date().toISOString()
    }
  },
  {
    id: '3',
    homeTeam: {
      id: 'team5',
      name: 'Bayern Munich',
      logoUrl: '/placeholder.svg',
      code: 'BAY'
    },
    awayTeam: {
      id: 'team6',
      name: 'Borussia Dortmund',
      logoUrl: '/placeholder.svg',
      code: 'DOR'
    },
    league: {
      id: 'league3',
      name: 'Bundesliga',
      country: 'Germany',
      logoUrl: '/placeholder.svg'
    },
    startTime: new Date(Date.now() + 10800000).toISOString(),
    status: 'scheduled',
    prediction: {
      homeWinProbability: 0.60,
      drawProbability: 0.20,
      awayWinProbability: 0.20,
      status: 'ready',
      lastUpdated: new Date().toISOString()
    }
  },
  {
    id: '4',
    homeTeam: {
      id: 'team7',
      name: 'Juventus',
      logoUrl: '/placeholder.svg',
      code: 'JUV'
    },
    awayTeam: {
      id: 'team8',
      name: 'AC Milan',
      logoUrl: '/placeholder.svg',
      code: 'MIL'
    },
    league: {
      id: 'league4',
      name: 'Serie A',
      country: 'Italy',
      logoUrl: '/placeholder.svg'
    },
    startTime: new Date(Date.now() - 3600000).toISOString(),
    status: 'live',
    score: {
      home: 2,
      away: 1
    },
    prediction: {
      homeWinProbability: 0.70,
      drawProbability: 0.20,
      awayWinProbability: 0.10,
      status: 'ready',
      lastUpdated: new Date().toISOString()
    }
  },
  {
    id: '5',
    homeTeam: {
      id: 'team9',
      name: 'PSG',
      logoUrl: '/placeholder.svg',
      code: 'PSG'
    },
    awayTeam: {
      id: 'team10',
      name: 'Lyon',
      logoUrl: '/placeholder.svg',
      code: 'LYO'
    },
    league: {
      id: 'league5',
      name: 'Ligue 1',
      country: 'France',
      logoUrl: '/placeholder.svg'
    },
    startTime: new Date(Date.now() - 7200000).toISOString(),
    status: 'finished',
    score: {
      home: 3,
      away: 0
    },
    prediction: {
      homeWinProbability: 0.55,
      drawProbability: 0.25,
      awayWinProbability: 0.20,
      status: 'ready',
      lastUpdated: new Date().toISOString()
    }
  }
];

// Mock user
const MOCK_USER: User = {
  id: '1',
  name: 'John Doe',
  email: 'john@example.com',
  role: 'user',
  avatar: '/placeholder.svg'
};

/**
 * API Service - In a real app, these would make actual API calls
 * Currently returning mock data for development purposes
 */
export const API = {
  // Auth functions
  auth: {
    login: async (email: string, password: string) => {
      // Mock successful login
      return { success: true, user: MOCK_USER };
    },
    register: async (name: string, email: string, password: string) => {
      // Mock successful registration
      return { success: true, user: { ...MOCK_USER, name, email } };
    },
    logout: async () => {
      return { success: true };
    },
    resetPassword: async (email: string) => {
      return { success: true };
    }
  },

  // Match functions
  matches: {
    getMatches: async (): Promise<Match[]> => {
      // Simulate network delay
      await new Promise(resolve => setTimeout(resolve, 800));
      return MOCK_MATCHES;
    },
    getMatchById: async (id: string): Promise<Match | undefined> => {
      await new Promise(resolve => setTimeout(resolve, 500));
      return MOCK_MATCHES.find(match => match.id === id);
    },
    getUpcomingMatches: async (): Promise<Match[]> => {
      await new Promise(resolve => setTimeout(resolve, 800));
      return MOCK_MATCHES.filter(match => 
        match.status === 'scheduled' || match.status === 'live'
      );
    }
  },

  // User functions
  user: {
    getProfile: async (): Promise<User> => {
      return MOCK_USER;
    },
    updateProfile: async (updates: Partial<User>): Promise<User> => {
      return { ...MOCK_USER, ...updates };
    }
  }
};
