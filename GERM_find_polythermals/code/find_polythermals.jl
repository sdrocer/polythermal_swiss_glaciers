using CSV
using DataFrames
using Statistics
using StatsBase
using GLMakie

# skript to organise the 2016 GERM output (Huss & Fischer, 2016) in order to identify clusters of polythermal glaciers.
# pp1 : Ablation area
# pp2 : ELA
# pp3 : Accumulation area

# read directory
sgi_ids = readdir("/Users/janoschbeer/PhD_local/all_glaciers")[2:end] # exclude .ds Store file

# reshaping the sgi-id to equalise it to SGI2010 sgi-ids
function replace_and_erase(input_string::AbstractString)
    # Convert the first character to uppercase and the rest to lowercase
    first_char = uppercase(input_string[1])
    rest_chars = lowercase(input_string[2:end])
    
    # Replace all capital letters except the first with lowercase
    replaced_string = first_char * replace(rest_chars, r"[A-Z]" => x -> lowercase(x.match[1]))
    
    # Erase all occurrences of "n"
    erased_string = replace(replaced_string, "n" => "")
    
    return erased_string
end
sgi_ids_corrected = replace_and_erase.(sgi_ids)
sgi_ids_without_underscore = replace.(sgi_ids_corrected, "_" => "-") # replace all underscores "_" with a "-" to make it converge with the original sgi-ids of the SGI2010


# function to declare NaNs as 0s
function NaNs_to_zeros(vector)
    rows_with_NaNs      = findall(x -> x == NaN,vector)
    @show rows_with_NaNs
    vector[rows_with_NaNs] .= 0
    return vector
end

function filter_NaNs(vector)
    vector = filter(x -> x != -9999.0, vector)
    return vector
end

# example glacier
A10_F_01 = CSV.read("/Users/janoschbeer/PhD_local/all_glaciers/A10F_01/out/permaseries_pp1_scpastrun.dat", delim=" ",ignorerepeated=true, DataFrame)
ten = A10_F_01[!,"10m"]
ten_noNaNs = NaNs_to_zeros(ten)
s = median(ten_noNaNs)

# get median ice temperature of all glaciers (pastrun: 1962-2010)
function GetMedIceTemp(sgi_ids,point,depth)
    med_temp = Float64[]
    for sgi_id in sgi_ids
        try
            gl_df = CSV.read("/Users/janoschbeer/PhD_local/all_glaciers/$sgi_id/out/permaseries_" * "$point" * "_scpastrun.dat", delim=" ", ignorerepeated=true, DataFrame) # open dataset per glacier & read as df
            ice_temp = gl_df[!,"$depth"]         # get glacier-specific ice temperature based on depth that is provided
            ice_temp = filter_NaNs(ice_temp)     # filter out all NaN values (-9999.0)
            median_ice_temp = median(ice_temp)   # get glacier-specific median 10m ice temperature
            push!(med_temp,median_ice_temp) 
        catch
            median_ice_temp = 0
            push!(med_temp,median_ice_temp) 
        end
    end
    return med_temp
end

# compute for Ablation area point
pp1_med_10m = GetMedIceTemp(sgi_ids,"pp1","10m")
pp1_med_20m = GetMedIceTemp(sgi_ids,"pp1","20m")

# compute for ELA point
pp2_med_10m = GetMedIceTemp(sgi_ids,"pp2","10m")
pp2_med_20m = GetMedIceTemp(sgi_ids,"pp2","20m")

# compute for Accumulation area point
pp3_med_10m = GetMedIceTemp(sgi_ids,"pp3","10m")
pp3_med_20m = GetMedIceTemp(sgi_ids,"pp3","20m")

# create final dataframe containing the sgi_ids & the glacier-specific median 10m ice temperatures 
ablation_temps = DataFrame(SGI=sgi_ids_without_underscore,med_10m=pp1_med_10m,med_20m=pp1_med_20m)   # pp1: Ablation area
ELA_temps = DataFrame(SGI=sgi_ids_without_underscore,med_10m=pp2_med_10m,med_20m=pp2_med_20m)        # pp2: ELA
acc_temps = DataFrame(SGI=sgi_ids_without_underscore,med_10m=pp3_med_10m,med_20m=pp3_med_20m)        # pp3: Accumulation area

# export dataframe as CSV
CSV.write("ablation_temps.csv", ablation_temps)
CSV.write("ELA_temps.csv", ELA_temps)
CSV.write("acc_temps.csv", acc_temps)

# histograms -> plots a histogram of a given ice temperature dataset based on SGI-IDs 
function IceTempsHistogram(sgi_ids,point,depth,glaciers_to_mark)
    location = point == "pp1" ? "ablation area" : 
    point == "pp2" ? "ELA" : "accumulation area" 
    ice_temps = GetMedIceTemp(sgi_ids,point,depth)
    ice_temps = filter(x -> x < -0.01 && x > -10, ice_temps) # filter out NaNs / zero values

    IceTempsHistogram = Figure(resolution=(1800,1000),fontsize=38)
    ax = Axis(IceTempsHistogram[1,1],title="Modelled median¹ $depth ice temperatures in the $location",xlabel="$depth ice temperature [°C]",ylabel="Frequency")
    hist = hist!(ax,ice_temps,bins=60,normalization=:none,color=:values,strokewidth=1,strokecolor=:black)

    # find most populated bin for visualization purposes
    histogram = fit(Histogram, ice_temps, nbins=60)
    max_count = maximum(histogram.weights)

    # mean & stdev
    mean = Statistics.mean(ice_temps)
    @show mean
    # stdev = std(ice_temps)
    vlines!(ax,[mean]; color=:red, linewidth=3, linestyle=:dash)
    
    # annotations
    for (glacier,value) in glaciers_to_mark
        vlines!(ax, [value]; color=:black, linewidth=3)
        text!(string(glacier), position = (value+(0.012*minimum(ice_temps)),60), align = (:left, :center), color=:black,fontsize=25,rotation=pi/2)
    end

    Label(IceTempsHistogram[2,1],"¹On the basis of GERM modelling results (Huss & Fischer 2016), covering a time series between 1960-2010", halign=:left,fontsize=25)
    Label(IceTempsHistogram[3,1],"Marked glaciers: Alphubel South (AS), Hohlaubgletscher (HL), Corvatsch (CV), Chessjengletscher (CH), Milibachgletscher (MB), Sex Rouge (SR)", halign=:left,fontsize=25)
    hidedecorations!

    # save
    save("Hist_$depth" * "_IceTemp_$location.png", IceTempsHistogram)
end

function IceTempsHistogramNorm(sgi_ids,point,depth,glaciers_to_mark)
    location = point == "pp1" ? "ablation area" : 
    point == "pp2" ? "ELA" : "accumulation area" 
    ice_temps = GetMedIceTemp(sgi_ids,point,depth)
    ice_temps = filter(x -> x < -0.01 && x > -10, ice_temps) # filter out NaNs / zero values

    IceTempsHistogramNorm = Figure(resolution=(1600,800),fontsize=38)
    ax = Axis(IceTempsHistogramNorm[1,1],title="Modelled median¹ $depth ice temperatures in the $location",xlabel="$depth ice temperature [°C]",ylabel="Fraction of Probability")
    hist = hist!(ax,ice_temps,bins=60,normalization=:probability,color=:values,strokewidth=1,strokecolor=:black)

    # find most populated bin for visualization purposes
    histogram = fit(Histogram, ice_temps, nbins=60)
    max_count = maximum(histogram.weights)
    
    # annotations
    for (glacier,value) in glaciers_to_mark
        vlines!(ax, [value]; color=:black, linewidth=2)
        text!(string(glacier), position = (value-0.07,0.04), align = (:left, :center), color=:black,fontsize=25,rotation=pi/2)
    end

    Label(IceTempsHistogramNorm[2,1],"¹On the basis of GERM modelling results (Huss & Fischer 2016), covering a time series between 1960-2010", halign=:left,fontsize=25)
    Label(IceTempsHistogram[3,1],"Marked glaciers: Alphubel South (AS), Hohlaubgletscher (HL), Corvatsch (CV), Chessjengletscher (CH), Milibachgletscher (MB), Sex Rouge (SR)", halign=:left,fontsize=25)
    hidedecorations!

    # save
    save("HistNorm_$depth" * "_IceTemp_$location.png", IceTempsHistogramNorm)
end

# glaciers to be marked
glacier_to_mark_10m = Dict("AS"=>-2.98,"CV"=>-1.59,"HL"=>-1.96, "CH"=>-1.190, "SR"=>-0.53, "MB"=>-0.91)
glacier_to_mark_20m = Dict("AS"=>-1.23,"CV"=>-0.768,"HL"=>-0.94, "CH"=>-0.601, "SR"=>-0.28, "MB"=>-0.458)

# plot
IceTempsHistogram(sgi_ids,"pp1","10m",glacier_to_mark_10m)
IceTempsHistogramNorm(sgi_ids,"pp1","10m",glacier_to_mark_10m)
IceTempsHistogram(sgi_ids,"pp1","20m",glacier_to_mark_20m)
IceTempsHistogram(sgi_ids,"pp2","10m")
IceTempsHistogram(sgi_ids,"pp3","10m")